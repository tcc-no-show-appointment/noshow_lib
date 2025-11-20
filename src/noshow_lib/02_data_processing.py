import pandas as pd
from pathlib import Path


def load_and_process_data(raw_path, processed_path, config: dict = None):
    """
    Carrega os dados brutos, aplica pré-processamento inicial baseado
    no dicionário de configuração (já convertido do YAML).

    Parameters
    ----------
    raw_path : str | Path
        Caminho do CSV bruto.
    
    processed_path : str | Path
        Caminho onde o CSV processado será salvo.
    
    config : dict
        Dicionário vindo do YAML (ex: config["preprocessing"]).
    """
    
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)

    # Garante que o diretório existe
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Carrega dados brutos
    df = pd.read_csv(raw_path)

    # =====================================================================
    # 1. Tratamento de valores ausentes
    # =====================================================================
    handle_missing = None
    if isinstance(config, dict):
        handle_missing = config.get("handle_missing")

    if handle_missing:
        if handle_missing == "drop":
            df = df.dropna()

        elif handle_missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))

        elif handle_missing == "median":
            df = df.fillna(df.median(numeric_only=True))

        elif handle_missing == "none":
            pass  # não faz nada

        else:
            raise ValueError(f"Método inválido em handle_missing: {handle_missing}")

    # =====================================================================
    # 2. Renomear colunas
    # =====================================================================
    if isinstance(config, dict):
        rename_cfg = config.get("rename_columns")
        if rename_cfg:
            if isinstance(rename_cfg, dict):
                df = df.rename(columns=rename_cfg)
            else:
                raise TypeError("rename_columns deve ser um dicionário")

    # =====================================================================
    # 3. Ajuste de tipos de dados
    # =====================================================================
    if isinstance(config, dict):
        cast_cfg = config.get("cast_types")
        if cast_cfg:
            if isinstance(cast_cfg, dict):
                for col, dtype in cast_cfg.items():
                    try:
                        df[col] = df[col].astype(dtype)
                    except Exception as e:
                        print(f"[WARN] Não foi possível converter '{col}' → '{dtype}': {e}")
            else:
                raise TypeError("cast_types deve ser um dicionário")

    # =====================================================================
    # FINAL: salva arquivo processado
    # =====================================================================
    df.to_csv(processed_path, index=False)
    print(f"[OK] Dados processados e salvos em: {processed_path}")

    return df
