import pandas as pd
import joblib
from pathlib import Path


def predict(data_path, model_path, predictions_path, threshold: float = 0.5):
    """
    Realiza predição usando um modelo/pipeline treinado.
    
    Parameters
    ----------
    data_path : str | Path
        Caminho para o CSV contendo os dados com features.
    
    model_path : str | Path
        Caminho para o pipeline salvo (.joblib).
    
    predictions_path : str | Path
        Caminho onde o CSV de previsões será salvo.
    
    threshold : float
        Limiar usado para transformar probabilidade em classe.
    """

    # Converte tudo para Path (garante compatibilidade Windows/Linux)
    data_path = Path(data_path)
    model_path = Path(model_path)
    predictions_path = Path(predictions_path)

    # ===============================
    # 1. Carrega dados para predição
    # ===============================
    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")

    df = pd.read_csv(data_path)

    # Remove target caso exista
    target_candidates = ["No-show", "no_show", "target"]
    cols_to_drop = [col for col in target_candidates if col in df.columns]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    X_predict = df

    # ===============================
    # 2. Carrega modelo treinado
    # ===============================
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo: {e}")

    # ===============================
    # 3. Realiza predições
    # ===============================
    if not hasattr(pipeline, "predict_proba"):
        raise AttributeError("O modelo carregado não possui método predict_proba.")

    predictions_prob = pipeline.predict_proba(X_predict)[:, 1]
    predictions_label = (predictions_prob >= threshold).astype(int)

    predictions_df = pd.DataFrame({
        "prediction_probability": predictions_prob,
        f"prediction_label_{threshold}": predictions_label
    })

    # ===============================
    # 4. Salva arquivo de saída
    # ===============================
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)

    print(f"[OK] Previsões geradas e salvas em: {predictions_path}")

    return predictions_df
