import joblib
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Optional, Any

from .logger import setup_logger

logger = setup_logger("noshow_lib.model_inference")


def predict(
    models: Dict[str, Any],
    input_data: pd.DataFrame,
    config: Dict,
    output_path: Optional[Union[str, Path]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Executa inferência roteando cada registro ao modelo da sua specialty_group.

    Args:
        models: Dict {specialty_group: modelo carregado}. Use load_models() para carregar.
        input_data: DataFrame de entrada.
        config: Dicionário de configuração.
        output_path: Caminho opcional para salvar o resultado em CSV.
        thresholds: Dict {specialty_group: threshold}. Usa 0.5 como fallback se None ou ausente.

    Returns:
        pd.DataFrame: DataFrame com [IDs, specialty_group, probability, prediction],
                      na mesma ordem do input.
    """
    if not isinstance(models, dict) or not models:
        raise ValueError("'models' deve ser um dicionário não-vazio {specialty_group: model}.")
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("'input_data' deve ser um pandas.DataFrame.")

    if output_path:
        output_path = Path(output_path)

    thresholds = thresholds or {}

    # 1. Validação de schema
    from .data_handler import load_and_validate
    try:
        df = load_and_validate(input_data, config, mode="inference")
    except Exception as e:
        logger.error(f"Erro na validação do schema de inferência: {e}")
        raise

    logger.info(f"Dados recebidos e validados. Shape: {df.shape}")

    # 2. Feature engineering automático se necessário
    feature_list = config.get("model_specialty", {}).get("features") or []
    missing_features = [c for c in feature_list if c not in df.columns]

    if missing_features or "specialty_group" not in df.columns:
        logger.info("Features ausentes ou specialty_group não encontrado. Executando build_features()...")
        from .feature_engineering import build_features
        try:
            df = build_features(df, config)
        except Exception as e:
            logger.error(f"Erro ao executar build_features: {e}")
            raise

    if "specialty_group" not in df.columns:
        raise ValueError(
            "Coluna 'specialty_group' não encontrada após feature engineering. "
            "Verifique se o config e os dados estão corretos."
        )

    # 3. Preservar IDs e índice original para reordenar no final
    id_cols = config.get("schema", {}).get("id_columns", ["appointment_id", "patient_id"])
    found_ids = [c for c in id_cols if c in df.columns]
    df = df.reset_index(drop=True)
    df["_original_index"] = df.index

    # 4. Inferência por specialty_group
    specialty_col = df["specialty_group"].astype(str)
    present_specialties = specialty_col.unique().tolist()
    unknown = [s for s in present_specialties if s not in models]
    if unknown:
        logger.warning(
            f"Especialidades sem modelo treinado (serão ignoradas): {unknown}. "
            f"Modelos disponíveis: {list(models.keys())}"
        )

    results = []

    for specialty, df_group in df.groupby(specialty_col):
        specialty = str(specialty)
        if specialty not in models:
            continue

        model = models[specialty]
        threshold = thresholds.get(specialty, 0.5)

        missing = [c for c in feature_list if c not in df_group.columns]
        if missing:
            logger.warning(f"[{specialty}] Features ausentes no grupo: {missing}. Pulando.")
            continue

        X = df_group[feature_list].copy()

        cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        for col in cat_features:
            X[col] = X[col].astype(str).replace("nan", "MISSING").astype("category")

        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.error(f"[{specialty}] Erro na predição: {e}")
            continue

        preds = (probs >= threshold).astype(int)

        group_result = df_group[found_ids + ["_original_index"]].copy()
        group_result["specialty_group"] = specialty
        group_result["probability"] = probs
        group_result["prediction"] = preds
        results.append(group_result)

    if not results:
        raise RuntimeError("Nenhuma predição foi gerada. Verifique os modelos e os dados de entrada.")

    result_df = (
        pd.concat(results, ignore_index=True)
        .sort_values("_original_index")
        .drop(columns=["_original_index"])
        .reset_index(drop=True)
    )

    # 5. Salvar output
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path, index=False)
            logger.info(f"Resultados salvos em: {output_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar resultados: {e}")
            raise

    logger.info(f"Inferência concluída. {len(result_df)} predições geradas.")
    return result_df


def load_models(models_dir: Union[str, Path], config: Dict) -> Dict[str, Any]:
    """
    Carrega todos os modelos por especialidade salvos em models_dir.

    Procura por arquivos com o padrão 'lgbm__{specialty}.joblib'.

    Args:
        models_dir: Diretório onde os modelos foram salvos.
        config: Dicionário de configuração (não utilizado diretamente, reservado para extensões).

    Returns:
        Dict[str, Any]: {specialty_name_uppercase: model_object}
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Diretório de modelos não encontrado: {models_dir}")

    model_files = list(models_dir.glob("lgbm__*.joblib"))
    if not model_files:
        raise FileNotFoundError(
            f"Nenhum modelo encontrado em '{models_dir}'. "
            "Certifique-se de ter executado train_model() antes."
        )

    loaded: Dict[str, Any] = {}
    for path in model_files:
        # Extrai specialty do nome: lgbm__clinica_especializada.joblib → CLINICA_ESPECIALIZADA
        specialty_raw = path.stem.replace("lgbm__", "")
        specialty_key = specialty_raw.upper()
        try:
            loaded[specialty_key] = joblib.load(path)
            logger.info(f"Modelo carregado: {path.name} → '{specialty_key}'")
        except Exception as e:
            logger.warning(f"Não foi possível carregar {path.name}: {e}")

    logger.info(f"{len(loaded)} modelo(s) carregado(s): {list(loaded.keys())}")
    return loaded
