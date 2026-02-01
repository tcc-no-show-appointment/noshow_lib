import yaml
from pathlib import Path
from typing import Union, Dict
from .logger import setup_logger

logger = setup_logger("noshow_lib.config")

def load_config(path: Union[str, Path]) -> Dict:
    """
    Carrega um arquivo de configuração YAML.
    
    Args:
        path: Caminho para o arquivo YAML.
        
    Returns:
        Dict: Dicionário de configuração.
        
    Raises:
        FileNotFoundError: Se o arquivo não existir.
        yaml.YAMLError: Se o arquivo tiver erro de sintaxe.
    """
    config_path = Path(path)
    
    if not config_path.exists():
        logger.error(f"Arquivo de configuração não encontrado: {config_path}")
        raise FileNotFoundError(f"Configuração ausente em {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        if config is None:
            logger.warning(f"O arquivo {config_path} está vazio.")
            return {}
            
        logger.info(f"Configuração carregada com sucesso de {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Erro de sintaxe no arquivo YAML {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado ao carregar config de {config_path}: {e}")
        raise
