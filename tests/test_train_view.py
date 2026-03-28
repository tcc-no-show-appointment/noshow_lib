import pandas as pd
from pathlib import Path
from noshow_lib import load_config, load_and_validate, build_features, train_model, setup_logger

logger = setup_logger("test_train_view")

CSV_PATH = Path(__file__).parent.parent / "abs_consultas_medicas_HT_v2.csv"

def main():
    logger.info("INICIANDO TESTE: TREINAMENTO VIA CSV (50.000 linhas)")

    config_path = Path(__file__).parent.parent / "config" / "local.yaml"
    config = load_config(config_path)

    logger.info(f"Carregando CSV: {CSV_PATH.name}...")
    df_raw = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig")
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
