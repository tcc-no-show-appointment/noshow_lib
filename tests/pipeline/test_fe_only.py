import pandas as pd
from pathlib import Path
from noshow_lib import load_config, load_and_validate, build_features, setup_logger

logger = setup_logger("test_fe_only")

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "abs_consultas_medicas_HT_v2.parquet"

def main():
    logger.info("INICIANDO TESTE: APENAS ENGENHARIA DE FEATURES (PARQUET)")

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "local.yaml"
    config = load_config(config_path)

    logger.info(f"Lendo Parquet: {DATA_PATH.name}...")
    df_raw = pd.read_parquet(DATA_PATH).head(1000)
    logger.info(f"Dados brutos carregados: {df_raw.shape}")

    df_validated = load_and_validate(df_raw, config)
    df_features = build_features(df_validated, config)

    print("\n" + "="*50)
    print("RESUMO DA ENGENHARIA DE FEATURES")
    print("="*50)
    print(f"Shape Final: {df_features.shape}")
    print(f"Novas colunas geradas (ex): {[c for c in df_features.columns if 'rate' in c or 'sin' in c or 'cos' in c][:5]}")
    print(f"Exemplo de waiting_days: {df_features['waiting_days'].head().tolist()}")
    print("="*50)

if __name__ == "__main__":
    main()
