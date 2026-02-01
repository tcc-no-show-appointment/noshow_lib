import pandas as pd
import joblib
from pathlib import Path
from typing import Union, Dict, Optional, List
from pandas.errors import EmptyDataError, ParserError
from .logger import setup_logger

logger = setup_logger("noshow_lib.model_inference")

def predict(
    input_data: Union[str, Path, pd.DataFrame], 
    model_path: Union[str, Path], 
    config: Dict,
    output_path: Optional[Union[str, Path]] = None, 
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Executa a inferência de forma robusta para produção.
    
    Args:
        input_data: Dados de entrada (CSV ou DataFrame).
        model_path: Caminho para o modelo treinado (.joblib).
        config: Dicionário de configuração.
        output_path: Caminho opcional para salvar o resultado em CSV.
        threshold: Limite de probabilidade para classificação.
        
    Returns:
        pd.DataFrame: DataFrame com IDs, probabilidades e predições.
    """
    model_path = Path(model_path)
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
    try:
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
            logger.info(f"Dados recebidos via DataFrame. Shape: {df.shape}")
        else:
            data_path = Path(input_data)
            if not data_path.exists():
                raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")
            df = pd.read_csv(data_path)
            logger.info(f"Dados carregados de {data_path}. Shape: {df.shape}")

    except (EmptyDataError, ParserError, Exception) as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise

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

    # 6. Carregar Modelo
    if not model_path.exists():
        logger.error(f"Arquivo do modelo não encontrado: {model_path}")
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")

    try:
        model = joblib.load(model_path)
        logger.info(f"Modelo carregado com sucesso de {model_path}")
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo .joblib: {e}")
        raise RuntimeError(f"Falha no carregamento do modelo: {e}")

    # 7. Executar Predição
    try:
        probs = model.predict_proba(X_inference)[:, 1]
        preds = (probs >= threshold).astype(int)
        logger.info("Cálculo de probabilidades concluído.")
    except Exception as e:
        logger.error(f"Erro durante a execução da predição: {e}")
        raise RuntimeError(f"Falha na execução da inferência: {e}")

    # 8. Consolidar Resultados
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
