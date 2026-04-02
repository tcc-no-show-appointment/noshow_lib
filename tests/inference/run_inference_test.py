import pandas as pd
import yaml
import logging
from pathlib import Path
from noshow_lib import predict, load_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "local.yaml"

DATA_PATH = PROJECT_ROOT / "abs_consultas_medicas_HT_v2.parquet"

def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("--> Carregando amostra de dados do Parquet para teste de inferência...")
    df_input = pd.read_parquet(DATA_PATH).head(100)

    if df_input is None or df_input.empty:
        print("Erro: Nenhum dado encontrado para teste.")
        return

    models = load_models(PROJECT_ROOT / "models", config)
    if not models:
        print("Nenhum modelo encontrado. Execute train_model() primeiro.")
        return

    print(f"--> Modelos carregados: {list(models.keys())}")
    print("--> Iniciando Inferência...")

    try:
        df_results = predict(
            models=models,
            input_data=df_input,
            config=config,
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
