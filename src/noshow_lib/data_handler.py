import pandas as pd
import yaml
from pathlib import Path
import logging

# Configuração básica de log para exibir no console se não houver um logger global
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_validate(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    """
    Carrega configuração, valida a presença de colunas e aplica tipagem inicial.
    """
    logger.info(f"Iniciando carga e validacao. Arquivo de config: {config_path}")

    # 1. Carga do Config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Arquivo de configuracao YAML carregado com sucesso.")
    except Exception as e:
        logger.error(f"Falha ao ler o arquivo de configuracao: {e}")
        raise

    # 2. Validação de Colunas
    required = config.get("schema", {}).get("required_columns", [])
    logger.info(f"Verificando presenca de {len(required)} colunas obrigatorias.")

    missing = list(set(required) - set(df.columns))
    if missing:
        logger.error(f"Validacao falhou. Colunas ausentes no banco de dados: {missing}")
        raise ValueError(f"Colunas ausentes: {missing}")

    logger.info("Validacao de schema concluida: todas as colunas obrigatorias foram encontradas.")

    # 3. Processamento Inicial e Casting (Tipagem)
    df_proc = df.copy()
    proc_config = config.get("preprocessing", {})

    if "cast_types" in proc_config:
        cast_dict = proc_config["cast_types"]
        logger.info(f"Iniciando conversao de tipos (casting) para {len(cast_dict)} colunas.")

        for col, dtype in cast_dict.items():
            if col in df_proc.columns:
                try:
                    if "date" in dtype or "time" in dtype:
                        df_proc[col] = pd.to_datetime(df_proc[col])
                        logger.debug(f"Coluna '{col}' convertida para datetime.")
                    else:
                        df_proc[col] = df_proc[col].astype(dtype)
                        logger.debug(f"Coluna '{col}' convertida para {dtype}.")
                except Exception as e:
                    logger.warning(f"Nao foi possivel converter a coluna '{col}' para {dtype}: {e}")

    logger.info("Processamento inicial concluido com sucesso.")
    return df_proc