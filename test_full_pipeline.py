import pandas as pd
from pathlib import Path
from noshow_lib import (
    load_config, 
    load_and_validate, 
    build_features, 
    train_model, 
    predict,
    setup_logger
)
from config.db import db

logger = setup_logger("test_full_pipeline")

def main():
    logger.info("INICIANDO TESTE: PIPELINE COMPLETO (RAW -> FE -> TRAIN -> PREDICT)")
    
    # 1. Carregar Configuração
    # Consumindo o config.yaml externo (na raiz do projeto)
    config_path = Path(__file__).parent / "config" / "prod.yaml"
    config = load_config(config_path)
    
    # 2. Dados Brutos
    logger.info("Etapa 1: Extraindo dados brutos...")
    df_raw = db.query("SELECT * FROM tb_appointments_ht")
    
    # 3. Processamento e Features
    logger.info("Etapa 2: Validação e Engenharia de Features...")
    df_validated = load_and_validate(df_raw, config)
    df_features = build_features(df_validated, config)
    
    # 4. Treinamento
    logger.info("Etapa 3: Treinamento do modelo...")
    train_results = train_model(df_features, config)
    
    # 5. Inferência (Teste de consistência)
    logger.info("Etapa 4: Teste de Inferência com dados brutos...")
    # Pegamos uma amostra dos dados brutos originais para ver se a inferência reconstrói as features
    df_inference_input = df_raw.tail(10) 
    predictions = predict(
        input_data=df_inference_input,
        model_path=train_results['artifact_path'],
        config=config
    )
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETO EXECUTADO COM SUCESSO")
    print("="*50)
    print(f"Treino ROC-AUC:  {train_results['metrics']['roc_auc']:.4f}")
    print(f"Predições (head):\n{predictions[['patient_id', 'probability', 'prediction']].head(3)}")
    print("="*50)

if __name__ == "__main__":
    main()
