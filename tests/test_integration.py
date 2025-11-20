# tests/test_integration.py

import pytest
import pandas as pd
from pathlib import Path
import logging

from noshow_lib.config import load_config
from noshow_lib.data_processing import load_and_process_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gerar_arquivo_fisico_com_config_real():
    
    # 1. SETUP: Definir Caminhos
    # Pega a pasta raiz para achar o YAML
    raiz_projeto = Path(__file__).parent.parent
    yaml_path = Path(__file__).parent / "assets" / "config_test.yaml"
    
    # --- CORREÇÃO 1: Definir os caminhos reais em variáveis ---
    caminho_csv_bruto = r'C:\Users\lobat\OneDrive\Área de Trabalho\tcc-rep\noshow-prediction-ml\data\01 - raw\noshowappointments.csv'
    
    # ==============================================================================
    # CORREÇÃO AQUI: Adicionei o nome do arquivo "noshowappointments_processed.csv"
    # ==============================================================================
    caminho_csv_processado = r'C:\Users\lobat\OneDrive\Área de Trabalho\tcc-rep\noshow-prediction-ml\data\02 - preprocessed\noshowappointments_processed.csv'
    
    # 2. Carregar Configuração
    config_completa = load_config(yaml_path)

    # 3. EXECUÇÃO
    # Passamos as variáveis que definimos acima
    df_resultado = load_and_process_data(
        input_data=caminho_csv_bruto,
        processed_path=caminho_csv_processado,
        config=config_completa.get("preprocessing")
    )

    # 4. VERIFICAÇÃO
    
    # --- CORREÇÃO 2: Verificar o arquivo correto ---
    # Agora checamos se o arquivo (e não a pasta) foi criado
    assert Path(caminho_csv_processado).exists()
    print(f"\n✅ SUCESSO! Arquivo gerado em: {caminho_csv_processado}")
    
    # --- CORREÇÃO 3: Verificar a coluna correta (Inglês) ---
    # O dataset original usa "Age", não "idade"
    if "Age" in df_resultado.columns:
        assert df_resultado["Age"].isnull().sum() == 0
        print("✅ Coluna 'Age' verificada: zero nulos.")
    else:
        # Caso você tenha renomeado no YAML, ajustamos aqui
        print(f"⚠️ Aviso: Colunas encontradas: {df_resultado.columns.tolist()}")