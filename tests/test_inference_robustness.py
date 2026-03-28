import pandas as pd
import yaml
import joblib
from pathlib import Path

from noshow_lib.model_inference import predict, load_models

# Input SEM a coluna 'Status' (simulando inferência real antes da consulta)
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    config_path = PROJECT_ROOT / "config" / "local.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df_input = pd.DataFrame([input_data])
    print("--> DataFrame de entrada (SEM STATUS):")
    print(df_input.T)

    models = load_models(PROJECT_ROOT / "models", config)
    if not models:
        print("Nenhum modelo encontrado. Execute train_model() primeiro.")
        return

    try:
        print("\n--> Executando inferência...")
        result_df = predict(
            models=models,
            input_data=df_input,
            config=config,
        )

        print("\n" + "="*40)
        print("RESULTADO DA PREDIÇÃO (SEM STATUS)")
        print("="*40)
        print(result_df)

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
