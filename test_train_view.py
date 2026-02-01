import pandas as pd
from pathlib import Path
from noshow_lib import load_config, train_model, setup_logger
from config.db import db

logger = setup_logger("test_train_view")

def main():
    logger.info("INICIANDO TESTE: TREINAMENTO VIA VIEW (VW_NOSHOW_TRAINING_DATA)")
    
    # 1. Setup
    # Consumindo o config.yaml externo (na raiz do projeto)
    config_path = Path(__file__).parent / "config" / "prod.yaml"
    config = load_config(config_path)
    
    # 2. Extração de Dados da View (Já processados)
    logger.info("Buscando dados da vw_noshow_training_data...")
    df_view = db.query("SELECT * FROM vw_noshow_training_data")
    logger.info(f"Dados da view carregados: {df_view.shape}")
    
    # 3. Treinamento Direto
    # Como a view já tem as features, podemos pular build_features 
    # (ou rodar build_features e ele apenas validará que está tudo lá)
    logger.info("Iniciando treinamento direto...")
    try:
        results = train_model(df_view, config)
        
        print("\n" + "="*50)
        print("RESULTADO DO TREINAMENTO VIA VIEW")
        print("="*50)
        print(f"ROC-AUC: {results['metrics']['roc_auc']:.4f}")
        print(f"Modelo salvo em: {results['artifact_path']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Erro no treinamento: {e}")

if __name__ == "__main__":
    main()
