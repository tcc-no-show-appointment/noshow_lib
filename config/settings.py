import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env se ele existir
# O arquivo .env deve estar na raiz do projeto
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Define o ambiente global ("local" ou "prod")
# Pode ser sobrescrito por variável de ambiente
ENV = os.getenv("APP_ENV", "local")
