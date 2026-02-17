
import pandas as pd
import yaml
import joblib
from pathlib import Path
import sys

# Adiciona o diretório src ao path para importar os módulos
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from noshow_lib.model_inference import predict

# Input SEM a coluna 'Status' (Simulando inferência real antes da consulta)
input_data = {
    "BairroPaciente": "BELA VISTA",
    "CEPUnidadeAtendimento": "04617-015",
    "CidadePaciente": "SAO PAULO",
    "DataHoraConsulta": "2024-11-23T14:00:00",
    "EnderecoUnidadeAtendimento": "RUA VIEIRA DE MORAES",
    "Especialidade": "CARDIOLOGIA",
    "Idade": 62,
    "Marcacao": "2024-11-16T08:00:00",
    "Sexo": "F",
    # "Status": "Realizado",  <-- REMOVIDO PROPOSITALMENTE
    "TipoConvenio": "Enfermaria",
    "UnidadeAtendimento": "CAMPO BELO",
    "id": 5642903,
    "idUnicoPaciente": "ID369425000"
}

def main():
    # 1. Carregar Configuração
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Criar DataFrame
    df_input = pd.DataFrame([input_data])
    print("--> DataFrame de entrada (SEM STATUS):")
    print(df_input.T)
    
    # 3. Carregar Modelo
    model_path = PROJECT_ROOT / "models" / "catboost_champion.joblib"
    if not model_path.exists():
        print(f"❌ Erro: Modelo não encontrado em {model_path}")
        return

    print(f"--> Carregando modelo de {model_path}...")
    model = joblib.load(model_path)

    # 4. Rodar Inferência
    try:
        print("\n--> Executando inferência...")
        result_df = predict(
            model=model,
            input_data=df_input,
            config=config
        )

        print("\n" + "="*40)
        print("RESULTADO DA PREDIÇÃO (SEM STATUS)")
        print("="*40)
        print(result_df)
        
        # Validação das features geradas
        # Se 'no_show_rate_patient' estiver presente, deve ser -1.0 (Cold Start)
        print("\nVerificação de Features Históricas (Cold Start):")
        # Para verificar isso, precisaríamos acessar o DF intermediário, mas vamos confiar no resultado final por enquanto.
        
        if not result_df.empty:
            proba = result_df.iloc[0]['probability']
            pred = result_df.iloc[0]['prediction']
            print(f"\nProbabilidade No-Show: {proba:.4f}")
            print(f"Predição: {pred}")
        print("="*40)

    except Exception as e:
        print(f"\nErro durante a inferência: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
