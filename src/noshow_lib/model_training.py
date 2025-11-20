import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
from category_encoders import TargetEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

def temporal_split(df, target_col, ts_col='AppointmentDay', ratios=(0.7, 0.3)):
    """
    Divide o dataframe em treino e teste com base em uma coluna de data/hora.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    
    n = len(df)
    n_train = int(n * ratios[0])
    
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test

def train_model(data_path, model_output_path, metrics_output_path, config):
    """
    Carrega os dados, treina o modelo e salva os artefatos.
    """
    # Carregar dados
    df = pd.read_csv(data_path)
    TARGET = config["data"]["target_column"]

    
    # Divisão temporal
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = temporal_split(df, TARGET)

    # Remover colunas de data/hora que não são features para o modelo
    datetime_cols = X_train.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns
    X_train = X_train.drop(columns=datetime_cols)
    X_test = X_test.drop(columns=datetime_cols)

    # Identificar tipos de features
    num_features = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
    cat_low = [c for c in cat_features if X_train[c].nunique() < 10]
    cat_high = [c for c in cat_features if X_train[c].nunique() >= 10]

    # Pipeline de pré-processamento
    transformers = [
        ("num", StandardScaler(), num_features),
        ("cat_low", OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_low),
        ("cat_high", TargetEncoder(handle_unknown='value'), cat_high)
    ]
    preprocessor = ColumnTransformer(transformers, remainder="passthrough")

    # Modelo
    lgbm = LGBMClassifier(
        n_estimators=300, learning_rate=0.01, num_leaves=20, max_depth=5,
        class_weight='balanced', subsample=0.7, colsample_bytree=0.7,
        n_jobs=-1, random_state=42
    )

    # Pipeline completo com SMOTE
    pipeline = ImbPipeline([
        ('preprocess', preprocessor),
        ('sampler', SMOTE(sampling_strategy=0.5, random_state=42)),
        ('clf', lgbm)
    ])

    # Treinamento
    pipeline.fit(X_train, y_train)

    # Avaliação
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "f1_score": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred)
    }
    print(f"Métricas de avaliação: {metrics}")

    # Salvar modelo e métricas
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"Modelo salvo em: {model_output_path}")

    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_output_path, index=False)
    print(f"Métricas salvas em: {metrics_output_path}")

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent
    features_data_path = project_dir / 'data' / '03 - features' / 'noshowappointments_features.csv'
    model_path = project_dir / 'models' / 'lgbm_pipeline.joblib'
    metrics_path = project_dir / 'data' / '04 - predictions' / 'lgbm_metrics.csv'

    # Carregar configuração
    config_path = project_dir / 'config' / 'local.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_model(features_data_path, model_path, metrics_path, config)
