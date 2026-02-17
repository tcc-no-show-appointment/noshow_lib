import pandas as pd
import joblib
from pathlib import Path
from typing import Union, Dict, Optional, List, Any
from pandas.errors import EmptyDataError, ParserError
from .logger import setup_logger

logger = setup_logger("noshow_lib.model_inference")

def predict(
    model: Any,
    input_data: pd.DataFrame, 
    config: Dict,
    output_path: Optional[Union[str, Path]] = None, 
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Executa a inferência de forma robusta para produção.
    
    Args:
        model: Objeto do modelo treinado (já carregado via joblib).
        input_data: Dados de entrada (DataFrame).
        config: Dicionário de configuração.
        output_path: Caminho opcional para salvar o resultado em CSV.
        threshold: Limite de probabilidade para classificação.
        
    Returns:
        pd.DataFrame: DataFrame com IDs, probabilidades e predições.
    """
    if output_path:
        output_path = Path(output_path)

    # 1. Carregar Configurações
    model_cfg = config.get("model", {})
    feature_list = model_cfg.get("features")
    if not feature_list:
        logger.error("A lista 'features' não foi encontrada no config.yaml")
        raise ValueError("A lista 'features' não foi encontrada no config.yaml")

    # Identificadores para manter no output
    id_cols = config.get("schema", {}).get("id_columns", ["appointment_id", "patient_id"])
    
    # 2. Carregar Dados
    # Assumindo que input_data já é um DataFrame conforme solicitado
    if not isinstance(input_data, pd.DataFrame):
         raise ValueError("input_data deve ser um pandas.DataFrame")
    
    # Validação de Schema de Inferência
    from .data_handler import load_and_validate
    try:
        df = load_and_validate(input_data, config, mode="inference")
    except Exception as e:
        logger.error(f"Erro na validação do schema de inferência: {e}")
        raise

    logger.info(f"Dados recebidos e validados. Shape: {df.shape}")

    # 3. Preservar Identificadores
    found_ids = [col for col in id_cols if col in df.columns]
    df_ids = df[found_ids].copy()
    if not found_ids:
        logger.warning(f"Nenhum dos IDs configurados {id_cols} foi encontrado no input.")

    # 4. Validação de Features (Feature Matching)
    missing_features = [col for col in feature_list if col not in df.columns]
    if missing_features:
        logger.info(f"Features ausentes no input: {missing_features}. Iniciando Feature Engineering...")
        from .feature_engineering import build_features
        try:
            df = build_features(df, config)
            # Re-verificar IDs após FE (pois FE pode filtrar linhas ou reordenar)
            found_ids = [col for col in id_cols if col in df.columns]
            df_ids = df[found_ids].copy()
            
            missing_features = [col for col in feature_list if col not in df.columns]
            if missing_features:
                error_msg = f"Inconsistência fatal. Colunas ainda ausentes após FE: {missing_features}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Erro ao tentar reconstruir features: {e}")
            raise

    # Selecionar apenas as features necessárias para o modelo (Allowlist)
    X_inference = df[feature_list].copy()
    logger.info(f"Features selecionadas para inferência: {len(feature_list)}")

    # 5. Consistência de Pré-processamento (Categóricas)
    cat_features = X_inference.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    if cat_features:
        logger.info(f"Tratando {len(cat_features)} colunas categóricas (Nulos -> 'MISSING').")
        for col in cat_features:
            X_inference[col] = X_inference[col].fillna('MISSING').astype(str)

    # 6. Executar Predição (Modelo já carregado)
    try:
        # Verifica se o modelo tem predict_proba, caso contrário usa predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_inference)[:, 1]
        else:
            # Fallback para modelos sem predict_proba (ex: regressão ou SVM simples), embora improvável aqui
            logger.warning("Modelo não possui predict_proba. Usando predict e assumindo 0/1.")
            probs = model.predict(X_inference)
            
        preds = (probs >= threshold).astype(int)
        logger.info("Cálculo de probabilidades concluído.")
    except Exception as e:
        logger.error(f"Erro durante a execução da predição: {e}")
        raise RuntimeError(f"Falha na execução da inferência: {e}")

    # 7. Consolidar Resultados

    result_df = df_ids.copy()
    result_df["probability"] = probs
    result_df["prediction"] = preds

    # 9. Salvar Output
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path, index=False)
            logger.info(f"Resultados de inferência salvos em: {output_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar arquivo de resultados: {e}")
            raise

    return result_df
