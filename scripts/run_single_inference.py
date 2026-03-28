import pandas as pd
import yaml
from pathlib import Path
from noshow_lib import predict, load_models

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Registro de exemplo (sem Status — inferência real antes da consulta)
input_data = {
    "BairroPaciente": "VILA PRUDENTE",
    "CEPUnidadeAtendimento": "03178-200",
    "CidadePaciente": "SAO PAULO",
    "DataHoraConsulta": "2024-12-10T09:00:00",
    "EnderecoUnidadeAtendimento": "AV SAPOPEMBA - SAO PAULO",
    "Especialidade": "FISIOTERAPIA",
    "Idade": 34,
    "Marcacao": "2024-11-20T10:00:00",
    "Sexo": "M",
    "TipoConvenio": "Particular",
    "UnidadeAtendimento": "VILA PRUDENTE",
    "id": 9900001,
    "idUnicoPaciente": "ID000001000"
}

def main():
    config_path = PROJECT_ROOT / "config" / "local.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df_input = pd.DataFrame([input_data])
    print("--> DataFrame de entrada:")
    print(df_input.T)

    models = load_models(PROJECT_ROOT / "models", config)
    if not models:
        print("Nenhum modelo encontrado. Execute train_model() primeiro.")
        return

    print(f"\n--> Modelos carregados: {list(models.keys())}")
    print("--> Executando inferência...")

    try:
        result_df = predict(
            models=models,
            input_data=df_input,
            config=config,
        )

        print("\n" + "="*40)
        print("RESULTADO DA PREDIÇÃO")
        print("="*40)
        print(result_df)

        if not result_df.empty:
            proba = result_df.iloc[0]['probability']
            pred = result_df.iloc[0]['prediction']
            print(f"\nProbabilidade No-Show: {proba:.4f}")
            print(f"Predição: {pred} ({'Falta' if pred == 1 else 'Comparece'})")
        print("="*40)

    except Exception as e:
        print(f"\nErro durante a inferência: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
