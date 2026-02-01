import sys
import yaml
import pandas as pd
import logging
from pathlib import Path

# Configuração de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ajuste de caminhos
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from config.db import db
    from src.noshow_lib.model_inference import predict
    print("--> Módulos importados com sucesso!")
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    sys.exit(1)

CONFIG_PATH = PROJECT_ROOT / "src" / "noshow_lib" / "config.yaml"
MODEL_PATH = PROJECT_ROOT / "models" / "catboost_champion.joblib"
OUTPUT_PATH = PROJECT_ROOT / "models" / "predictions_test.csv"

def main():
    # 1. Carregar Config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Buscar pequena amostra de dados
    print("--> Buscando amostra de dados brutos para teste de inferência...")
    query = "SELECT TOP 100 * FROM tb_appointments_ht"
    df_input = db.query(query)

    if df_input is None or df_input.empty:
        print("❌ Erro: Nenhum dado encontrado para teste.")
        return

    # 3. Executar Inferência
    print("--> Iniciando Inferência...")
    try:
        df_results = predict(
            input_data=df_input,
            model_path=MODEL_PATH,
            config=config,
            output_path=OUTPUT_PATH
        )
        
        print("\n" + "="*40)
        print("TESTE DE INFERÊNCIA CONCLUÍDO")
        print("="*40)
        print(f"Shape do resultado: {df_results.shape}")
        print(f"Colunas retornadas: {df_results.columns.tolist()}")
        print("\nPrimeiras 5 linhas:")
        print(df_results.head())
        print("="*40)

    except Exception as e:
        print(f"\nERRO DURANTE A INFERÊNCIA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
