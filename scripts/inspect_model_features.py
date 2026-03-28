import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

model_files = list(MODELS_DIR.glob("lgbm__*.joblib"))

if not model_files:
    print(f"Nenhum modelo encontrado em {MODELS_DIR}")
else:
    for model_path in sorted(model_files):
        print(f"\n{'='*50}")
        print(f"Modelo: {model_path.name}")
        print(f"{'='*50}")
        model = joblib.load(model_path)
        if hasattr(model, "feature_name_"):
            for i, name in enumerate(model.feature_name_):
                print(f"  {i+1:2}. {name}")
        elif hasattr(model, "feature_names_in_"):
            for i, name in enumerate(model.feature_names_in_):
                print(f"  {i+1:2}. {name}")
        else:
            print("  Modelo não possui atributo de nomes de features.")
