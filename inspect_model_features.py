import joblib
from pathlib import Path

model_path = Path("models/catboost_champion.joblib")
if model_path.exists():
    model = joblib.load(model_path)
    if hasattr(model, "feature_names_"):
        print("Features used in the model:")
        for i, name in enumerate(model.feature_names_):
            print(f"{i+1}. {name}")
    else:
        print("Model does not have feature_names_ attribute.")
else:
    print(f"Model file not found at {model_path}")
