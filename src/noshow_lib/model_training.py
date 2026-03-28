import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from .logger import setup_logger

logger = setup_logger("noshow_lib.model_training")

# Silencia logs verbosos do Optuna e LightGBM
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_model(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Treina um modelo LightGBM independente por specialty_group.

    Args:
        df: DataFrame processado com features (saída de build_features).
        config: Dicionário de configuração (YAML).

    Returns:
        Dict[str, Dict]: Um dicionário por specialty_group com modelo, métricas,
                         threshold ótimo e caminhos dos artefatos.
        Exemplo:
        {
            "CLINICA_ESPECIALIZADA": {
                "model": <LGBMClassifier>,
                "metrics": {...},
                "threshold": 0.35,
                "artifact_path": "models/lgbm__clinica_especializada.joblib",
                ...
            },
            ...
        }
    """
    model_cfg = config.get("model_specialty", {})
    feature_list = model_cfg.get("features")
    if not feature_list:
        raise ValueError("A lista 'features' não foi encontrada em config['model_specialty'].")

    target = config.get("data", {}).get("target_column", "no_show")
    date_col = config.get("split", {}).get("date_column", "appointment_at")
    artifact_dir = Path(model_cfg.get("artifact_dir", "models/"))
    min_volume = model_cfg.get("min_volume", 5000)
    optuna_trials = model_cfg.get("optuna_trials", 50)
    train_frac = model_cfg.get("train_frac", 0.70)
    val_frac = model_cfg.get("val_frac", 0.15)

    if "specialty_group" not in df.columns:
        raise ValueError("Coluna 'specialty_group' não encontrada. Execute build_features() antes.")
    if target not in df.columns:
        raise ValueError(f"Coluna alvo '{target}' não encontrada no DataFrame.")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}
    skipped = []

    specialties = sorted(df["specialty_group"].dropna().astype(str).unique())
    logger.info(f"Especialidades encontradas: {specialties}")

    for specialty in specialties:
        specialty_str = str(specialty)
        df_group = df[df["specialty_group"].astype(str) == specialty_str].copy()

        if len(df_group) < min_volume:
            logger.warning(
                f"[{specialty_str}] Volume insuficiente ({len(df_group)} < {min_volume}). Pulando."
            )
            skipped.append(specialty_str)
            continue

        logger.info(f"[{specialty_str}] Iniciando treinamento | {len(df_group)} registros...")
        try:
            result = _train_single_specialty(
                df_group=df_group,
                config=config,
                specialty_name=specialty_str,
                feature_list=feature_list,
                target=target,
                date_col=date_col,
                artifact_dir=artifact_dir,
                optuna_trials=optuna_trials,
                train_frac=train_frac,
                val_frac=val_frac,
            )
            results[specialty_str] = result
            logger.info(
                f"[{specialty_str}] Concluído | PR-AUC: {result['metrics']['pr_auc']:.4f} "
                f"| threshold: {result['threshold']:.2f}"
            )
        except Exception as e:
            logger.error(f"[{specialty_str}] Erro no treinamento: {e}")
            skipped.append(specialty_str)

    if not results:
        raise RuntimeError("Nenhuma especialidade foi treinada com sucesso.")

    # Métricas consolidadas
    consolidated = {
        specialty: {
            "pr_auc": r["metrics"]["pr_auc"],
            "roc_auc": r["metrics"]["roc_auc"],
            "f1": r["metrics"]["f1"],
            "recall": r["metrics"]["recall"],
            "precision": r["metrics"]["precision"],
            "threshold": r["threshold"],
            "artifact_path": r["artifact_path"],
        }
        for specialty, r in results.items()
    }
    consolidated_path = artifact_dir / "comparativo_especialidades.json"
    with open(consolidated_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=4, ensure_ascii=False)
    logger.info(f"Métricas consolidadas salvas em: {consolidated_path}")

    if skipped:
        logger.warning(f"Especialidades ignoradas: {skipped}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers privados
# ─────────────────────────────────────────────────────────────────────────────

def _chronological_split(
    df: pd.DataFrame,
    date_col: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split cronológico sem embaralhamento para evitar data leakage."""
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df_sorted.iloc[:train_end], df_sorted.iloc[train_end:val_end], df_sorted.iloc[val_end:]


def _build_xy(
    df: pd.DataFrame,
    feature_list: list,
    target: str,
    cat_features: list,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Seleciona features e trata categóricas."""
    X = df[feature_list].copy()
    y = df[target].copy()

    for col in cat_features:
        if col in X.columns:
            X[col] = X[col].astype(str).replace("nan", "MISSING").astype("category")

    num_cols = [c for c in feature_list if c not in cat_features]
    for col in num_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y


def _optuna_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_features: list,
) -> float:
    """Função objetivo do Optuna: maximiza PR-AUC na validação."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "n_estimators": 1000,
        "random_state": 42,
        "verbose": -1,
    }

    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
        categorical_feature=cat_features if cat_features else "auto",
    )
    y_proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_proba)


def _find_best_threshold(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Encontra o threshold que maximiza F1 na validação."""
    thresholds = np.arange(0.20, 0.71, 0.05)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return float(best_t)


def _train_single_specialty(
    df_group: pd.DataFrame,
    config: Dict,
    specialty_name: str,
    feature_list: list,
    target: str,
    date_col: str,
    artifact_dir: Path,
    optuna_trials: int,
    train_frac: float,
    val_frac: float,
) -> Dict:
    """Treina um modelo LightGBM para uma única especialidade."""

    # 1. Split cronológico
    train_df, val_df, test_df = _chronological_split(df_group, date_col, train_frac, val_frac)

    for split_name, split_df in [("treino", train_df), ("validação", val_df), ("teste", test_df)]:
        if split_df.empty:
            raise ValueError(f"Split de {split_name} está vazio para '{specialty_name}'.")
        classes = split_df[target].dropna().unique()
        if len(classes) < 2:
            raise ValueError(
                f"Split de {split_name} tem apenas uma classe ({classes}) para '{specialty_name}'."
            )

    # 2. Features e categóricas
    missing_cols = [c for c in feature_list if c not in df_group.columns]
    if missing_cols:
        raise ValueError(f"Features ausentes no DataFrame: {missing_cols}")

    cat_features = [
        c for c in feature_list
        if df_group[c].dtype in ("object", "category", "string")
        or str(df_group[c].dtype) == "string"
    ]

    X_train, y_train = _build_xy(train_df, feature_list, target, cat_features)
    X_val, y_val = _build_xy(val_df, feature_list, target, cat_features)
    X_test, y_test = _build_xy(test_df, feature_list, target, cat_features)

    # Alinhar colunas categóricas entre splits
    for col in cat_features:
        if col in X_train.columns:
            all_cats = pd.api.types.union_categoricals(
                [X_train[col], X_val[col], X_test[col]]
            ).categories
            X_train[col] = X_train[col].cat.set_categories(all_cats)
            X_val[col] = X_val[col].cat.set_categories(all_cats)
            X_test[col] = X_test[col].cat.set_categories(all_cats)

    logger.info(
        f"[{specialty_name}] Split: treino={len(X_train)}, val={len(X_val)}, teste={len(X_test)}"
    )

    # 3. Optuna
    logger.info(f"[{specialty_name}] Iniciando Optuna ({optuna_trials} trials)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train, X_val, y_val, cat_features),
        n_trials=optuna_trials,
        show_progress_bar=False,
    )
    best_params = study.best_params
    logger.info(f"[{specialty_name}] Melhor PR-AUC (val): {study.best_value:.4f}")

    # 4. Treino final
    best_params.update({"n_estimators": 3000, "random_state": 42, "verbose": -1})
    final_model = LGBMClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(100)],
        categorical_feature=cat_features if cat_features else "auto",
    )

    # 5. Threshold ótimo na validação
    y_proba_val = final_model.predict_proba(X_val)[:, 1]
    best_threshold = _find_best_threshold(y_val, y_proba_val)

    # 6. Métricas no teste
    y_proba_test = final_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= best_threshold).astype(int)

    metrics = {
        "pr_auc": float(average_precision_score(y_test, y_proba_test)),
        "roc_auc": float(roc_auc_score(y_test, y_proba_test)),
        "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred_test)),
        "volume_train": int(len(X_train)),
        "volume_val": int(len(X_val)),
        "volume_test": int(len(X_test)),
        "target_rate_test": float(y_test.mean()),
    }

    # 7. Salvar artefatos
    safe_name = specialty_name.lower().replace(" ", "_")
    artifact_path = artifact_dir / f"lgbm__{safe_name}.joblib"
    metrics_path = artifact_dir / f"lgbm__{safe_name}_metrics.json"

    joblib.dump(final_model, artifact_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({**metrics, "threshold": best_threshold, "best_params": best_params}, f, indent=4)

    logger.info(f"[{specialty_name}] Modelo salvo em: {artifact_path}")

    return {
        "model": final_model,
        "metrics": metrics,
        "threshold": best_threshold,
        "artifact_path": str(artifact_path),
        "metrics_path": str(metrics_path),
        "features_used": feature_list,
        "cat_features": cat_features,
        "best_params": best_params,
    }
