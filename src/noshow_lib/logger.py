import logging
import sys
from pathlib import Path

def setup_logger(name: str = "noshow_lib", level: int = logging.INFO) -> logging.Logger:
    """
    Configura um logger padronizado para a biblioteca.
    """
    logger = logging.getLogger(name)
    
    # Evita duplicar handlers se o logger já estiver configurado
    if not logger.handlers:
        logger.setLevel(level)
        
        # Formato do log: Data - Nome - Level - Mensagem
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Opcional: File Handler (pode ser ativado via config se necessário)
        # log_file = Path("noshow_lib.log")
        # file_handler = logging.FileHandler(log_file)
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)
        
    return logger

# Instância padrão para uso rápido
logger = setup_logger()
