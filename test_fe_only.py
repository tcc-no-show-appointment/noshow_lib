import pandas as pd
from pathlib import Path
from noshow_lib import load_config, load_and_validate, build_features, setup_logger
from config.db import db

logger = setup_logger("test_fe_only")

def main():
    logger.info("INICIANDO TESTE: APENAS ENGENHARIA DE FEATURES (TABELA BRUTA)")
    
    # 1. Setup
    # Consumindo o config.yaml externo (na raiz do projeto)
    config_path = Path(__file__).parent / "config" / "prod.yaml"
    

    config = load_config(config_path)
    
    # 2. Extração de Dados Brutos
    logger.info("Buscando dados da tb_appointments_ht...")
    df_raw = db.query("SELECT TOP 1000 * FROM tb_appointments_ht")
    logger.info(f"Dados brutos carregados: {df_raw.shape}")
    
    # 3. Validação e Tipagem
    df_validated = load_and_validate(df_raw, config)
    
    # 4. Feature Engineering
    df_features = build_features(df_validated, config)
    
    # 5. Verificação de Resultados
    print("\n" + "="*50)
    print("RESUMO DA ENGENHARIA DE FEATURES")
    print("="*50)
    print(f"Shape Final: {df_features.shape}")
    print(f"Novas colunas geradas (ex): {[c for c in df_features.columns if 'rate' in c or 'sin' in c or 'cos' in c][:5]}")
    print(f"Exemplo de waiting_days: {df_features['waiting_days'].head().tolist()}")
    print("="*50)

if __name__ == "__main__":
    main()
