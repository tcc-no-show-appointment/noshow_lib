import os
import json
import joblib
import pandas as pd
from typing import Dict, Any
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from .logger import setup_logger

logger = setup_logger("noshow_lib.model_training")

def train_model(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Treina o modelo CatBoost usando as configurações do YAML.
    
    Args:
        df: DataFrame processado com as features.
        config: Dicionário de configuração.
        
    Returns:
        Dict: Artefatos do treinamento (modelo, métricas, caminhos).
    """
    # 1. Extração de Configurações
    model_cfg = config.get("model", {})
    params = model_cfg.get("parameters", {}).copy() # Cópia para não mutar o original
    split_cfg = config.get("split", {})
    target = config.get("data", {}).get("target_column", "no_show")
    artifact_path = model_cfg.get("artifact_path", "models/model.joblib")
    feature_list = model_cfg.get("features")
    
    if not feature_list:
        logger.error("A lista 'features' não foi encontrada no config.yaml")
        raise ValueError("A lista 'features' é obrigatória no config.yaml")

    # 2. Split de Dados (Estratificado)
    logger.info("Iniciando preparação do split de treino/teste...")
    test_frac = split_cfg.get("test_frac", 0.2)
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_frac, 
        random_state=42, 
        stratify=df[target] if target in df.columns else None
    )
    
    # 3. Seleção de Features e Target
    missing_cols = [col for col in feature_list if col not in train_df.columns]
    if missing_cols:
        error_msg = f"Features configuradas ausentes no DataFrame: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    X_train = train_df[feature_list].copy()
    X_test = test_df[feature_list].copy()
    y_train = train_df[target]
    y_test = test_df[target]

    # 4. Tratamento de Variáveis Categóricas
    cat_features = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    logger.info(f"Features categóricas detectadas: {cat_features}")

    for col in cat_features:
        X_train[col] = X_train[col].fillna('MISSING').astype(str)
        X_test[col] = X_test[col].fillna('MISSING').astype(str)

    # 5. Treinamento
    logger.info(f"Iniciando treinamento do CatBoost ({len(X_train)} amostras)...")
    
    # Extração de parâmetros de controle do fit
    early_stopping = params.pop("early_stopping_rounds", 50)
    
    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        early_stopping_rounds=early_stopping,
        verbose=params.get("verbose", 100)
    )

    # 6. Avaliação
    logger.info("Calculando métricas de desempenho...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision_class_1": float(classification_report(y_test, y_pred, output_dict=True)['1']['precision']),
        "recall_class_1": float(classification_report(y_test, y_pred, output_dict=True)['1']['recall']),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # 7. Salvamento de Artefatos
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(model, artifact_path)
    
    metrics_path = artifact_path.replace(".joblib", ".json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Treinamento finalizado. Modelo: {artifact_path} | AUC: {metrics['roc_auc']:.4f}")
    
    return {
        "model": model,
        "metrics": metrics,
        "artifact_path": artifact_path,
        "metrics_path": metrics_path,
        "features_used": feature_list,
        "cat_features": cat_features
    }
