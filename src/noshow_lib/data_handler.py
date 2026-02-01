import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Union, Optional
from .logger import setup_logger

logger = setup_logger("noshow_lib.data_handler")

def load_and_validate(df: pd.DataFrame, config: Union[str, Path, Dict]) -> pd.DataFrame:
    """
    Valida a presença de colunas obrigatórias e aplica tipagem inicial.
    
    Args:
        df: DataFrame a ser validado.
        config: Caminho para o arquivo YAML ou dicionário de configuração.
        
    Returns:
        pd.DataFrame: DataFrame validado e com tipos convertidos.
    """
    # 1. Resolver Configuração
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = config

    logger.info("Iniciando validação de dados...")

    # 2. Validação de Schema (Colunas Obrigatórias)
    required_columns = config_dict.get("schema", {}).get("required_columns", [])
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Falha na validação de schema. Colunas ausentes: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"Sucesso: Todas as {len(required_columns)} colunas obrigatórias foram encontradas.")

    # 3. Conversão de Tipos (Casting)
    df_processed = df.copy()
    preprocessing_cfg = config_dict.get("preprocessing", {})
    cast_types = preprocessing_cfg.get("cast_types", {})

    if cast_types:
        logger.info(f"Aplicando conversão de tipos para {len(cast_types)} colunas.")
        for col, dtype in cast_types.items():
            if col in df_processed.columns:
                try:
                    if any(date_term in dtype.lower() for date_term in ["date", "time", "datetime"]):
                        df_processed[col] = pd.to_datetime(df_processed[col], errors="coerce")
                    else:
                        df_processed[col] = df_processed[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Não foi possível converter a coluna '{col}' para {dtype}: {e}")

    logger.info("Carga e validação concluídas com sucesso.")
    return df_processed
