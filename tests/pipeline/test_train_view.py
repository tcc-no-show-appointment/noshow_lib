import pandas as pd
from pathlib import Path
from noshow_lib import load_config, load_and_validate, build_features, train_model, setup_logger

logger = setup_logger("test_train_view")

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "abs_consultas_medicas_HT_v2.parquet"

def main():
    logger.info("INICIANDO TESTE: TREINAMENTO VIA PARQUET")

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "local.yaml"
    config = load_config(config_path)

    logger.info(f"Carregando Parquet: {DATA_PATH.name}...")
    df_raw = pd.read_parquet(DATA_PATH)
    logger.info(f"Dados carregados: {df_raw.shape}")

    df_validated = load_and_validate(df_raw, config)
    df_view = build_features(df_validated, config)
    logger.info(f"Features geradas: {df_view.shape}")

    logger.info("Iniciando treinamento por especialidade...")
    try:
        results = train_model(df_view, config)

        print("\n" + "=" * 50)
        print("RESULTADO DO TREINAMENTO VIA VIEW")
        print("=" * 50)
        for name, r in results.items():
            print(
                f"  {name}: PR-AUC={r['metrics']['pr_auc']:.4f} "
                f"| ROC-AUC={r['metrics']['roc_auc']:.4f} "
                f"| threshold={r['threshold']:.2f} "
                f"| modelo={r['artifact_path']}"
            )
        print("=" * 50)

    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")

if __name__ == "__main__":
    main()
