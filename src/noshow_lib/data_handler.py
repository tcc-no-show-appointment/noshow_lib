import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Union, Optional
from .logger import setup_logger

logger = setup_logger("noshow_lib.data_handler")

def load_and_validate(df: pd.DataFrame, config: Union[str, Path, Dict], mode: str = "training") -> pd.DataFrame:
    """
    Valida a presença de colunas obrigatórias e aplica tipagem inicial.
    
    Args:
        df: DataFrame a ser validado.
        config: Caminho para o arquivo YAML ou dicionário de configuração.
        mode: Modo de operação ("training" ou "inference"). Define qual lista de colunas validar.
        
    Returns:
        pd.DataFrame: DataFrame validado e com tipos convertidos.
    """
    # 1. Resolver Configuração
    if isinstance(config, (str, Path, dict)):
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = config
    else:
        raise ValueError("Configuração inválida. Deve ser um caminho (str/Path) ou um dicionário.")

    logger = setup_logger("noshow_lib.data_handler")
    logger.info(f"Iniciando validação de dados no modo: {mode}...")

    # 2. Validação de Schema (Colunas Obrigatórias)
    # Seleciona a lista correta baseada no modo
    schema_config = config_dict.get("schema", {})
    
    if mode == "training":
        # Tenta pegar 'training_columns', se não existir, tenta o antigo 'required_columns'
        required_columns = schema_config.get("training_columns") or schema_config.get("required_columns", [])
    elif mode == "inference":
        required_columns = schema_config.get("inference_columns", [])
    else:
        logger.warning(f"Modo desconhecido '{mode}'. Usando validação padrão (inference).")
        required_columns = schema_config.get("inference_columns", [])
    
    # Valida se as colunas existem no DataFrame
    if required_columns:
        # Verifica quais colunas da lista não estão presentes no DataFrame
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f"Falha na validação de schema ({mode}). Colunas ausentes: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Sucesso: Todas as {len(required_columns)} colunas obrigatórias foram encontradas.")
    else:
        logger.warning(f"Nenhuma coluna obrigatória configurada para o modo '{mode}'.")

    # 3. Conversão de Tipos (Casting)
    df_processed = df.copy()
    
    # Casting opcional para colunas numéricas (apenas se existirem e não forem nulas)
    # Isso evita erro ao tentar converter 'Status' (agora opcional) se ele não existir
    preprocessing_cfg = config_dict.get("preprocessing", {})
    cast_types = preprocessing_cfg.get("cast_types", {})

    if cast_types:
        logger.info(f"Aplicando conversão de tipos para {len(cast_types)} colunas.")
        for col, dtype in cast_types.items():
            if col in df_processed.columns:
                try:
                    # Verifica se a coluna tem valores antes de converter (evita erros em colunas vazias)
                    if df_processed[col].notna().any():
                        if any(date_term in dtype.lower() for date_term in ["date", "time", "datetime"]):
                            df_processed[col] = pd.to_datetime(df_processed[col], errors="coerce")
                        else:
                            df_processed[col] = df_processed[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Não foi possível converter a coluna '{col}' para {dtype}: {e}")

    logger.info("Carga e validação concluídas com sucesso.")
    return df_processed
