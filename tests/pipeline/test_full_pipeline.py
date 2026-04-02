import pandas as pd
from pathlib import Path
from noshow_lib import (
    load_config,
    load_and_validate,
    build_features,
    train_model,
    predict,
    load_models,
    setup_logger
)

logger = setup_logger("test_full_pipeline")

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "abs_consultas_medicas_HT_v2.parquet"

def main():
    logger.info("INICIANDO TESTE: PIPELINE COMPLETO (PARQUET -> FE -> TRAIN -> PREDICT)")

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "local.yaml"
    config = load_config(config_path)

    logger.info("Etapa 1: Carregando dados do Parquet (amostra de 20.000 linhas)...")
    df_raw = pd.read_parquet(DATA_PATH).head(20_000)

    logger.info("Etapa 2: Validação e Engenharia de Features...")
    df_validated = load_and_validate(df_raw, config)
    df_features = build_features(df_validated, config)

    logger.info("Etapa 3: Treinamento por especialidade...")
    train_results = train_model(df_features, config)

    logger.info("Etapa 4: Teste de Inferência...")
    models = load_models(config["model_specialty"]["artifact_dir"], config)
    thresholds = {name: r["threshold"] for name, r in train_results.items()}

    df_inference_input = df_raw.tail(100)
    predictions = predict(
        models=models,
        input_data=df_inference_input,
        config=config,
        thresholds=thresholds,
    )

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETO EXECUTADO COM SUCESSO")
    print("=" * 50)
    print(f"Especialidades treinadas: {list(train_results.keys())}")
    for name, r in train_results.items():
        print(f"  {name}: PR-AUC={r['metrics']['pr_auc']:.4f} | threshold={r['threshold']:.2f}")
    print(f"\nPredições (head):\n{predictions.head(5)}")
    print("=" * 50)

if __name__ == "__main__":
    main()
