import os
import sys
import logging
import pandas as pd

# Adiciona o diretório 'src' ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from noshow_lib.config import load_config
from noshow_lib.data_handler import load_and_validate
from noshow_lib.feature_engineering import build_features
from config.db import Database

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    """Função principal para executar o teste do pipeline."""
    try:
        # Carregar configurações
        config_path = "src/noshow_lib/config.yaml"
        config = load_config(config_path)

        # Conectar ao banco e carregar dados
        db = Database()
        query = "SELECT TOP 1000 * FROM tb_appointments_ht"  # Limita para teste rápido
        df_raw = db.query(query)

        if df_raw.empty:
            logging.warning("O DataFrame carregado do banco de dados está vazio.")
            return

        # Validar e pré-processar os dados
        df_validated = load_and_validate(df_raw, config_path=config_path)
        print("Dados carregados, validados e tipados com sucesso.")

        # Aplicar engenharia de features
        df_featured = build_features(df_validated, config=config)

        # Imprimir informações para verificação
        print("\n--- INFORMAÇÕES DO DATAFRAME PROCESSADO ---")
        print("\nShape final:", df_featured.shape)
        print("\n5 primeiras linhas:")
        print(df_featured.head())
        print("\nInformações do DataFrame:")
        df_featured.info()

    except Exception as e:
        logging.error(f"Falha no pipeline de dados: {e}", exc_info=True)
        print(f"\n--- ERRO NO PIPELINE ---\nFalha no pipeline de dados: {e}")

if __name__ == "__main__":
    main()
