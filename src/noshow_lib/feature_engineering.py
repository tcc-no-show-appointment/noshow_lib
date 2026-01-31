import pandas as pd
import numpy as np
import logging
import holidays
import re
from typing import Dict

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Orquestra as transformacoes do notebook usando .pipe().
    Recebe o config para mapeamento dinamico de colunas.
    """
    logger.info("Iniciando Pipeline de Feature Engineering...")

    df_processed = (
        df.copy()
        .pipe(_rename_columns, config)
        .pipe(_filter_noshow_status)
        .pipe(_convert_initial_types)
        .pipe(_create_target_variable)
        .pipe(_create_temporal_features)
        .pipe(_create_patient_history)
        .pipe(_create_age_groups)
        .pipe(_create_behavioral_features) # Adicionado
        .pipe(_create_contextual_rates) # Adicionado
        .pipe(create_holiday_feature) # Adicionado
        .pipe(_normalize_text)
        .pipe(_reorder_columns, config) # Passando o config
    )

    logger.info(f"Feature engineering finalizado. Shape: {df_processed.shape}")
    return df_processed


def _rename_columns(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Aplica o mapeamento do YAML e normaliza para snake_case."""
    # O YAML pode ter uma estrutura aninhada, ex: data.column_mapping
    column_map = config.get("column_map", {})
    df = df.rename(columns=column_map)
    df.columns = [re.sub(r'[^0-9a-zA-Z_]+', '_', c).lower().strip('_') for c in df.columns]
    logger.info(f"Colunas renomeadas e normalizadas. Ex: {list(df.columns[:3])}")
    return df


def _filter_noshow_status(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas registros com desfecho binário conhecido."""
    if "appointment_status" not in df.columns:
        raise KeyError("Coluna 'appointment_status' não encontrada. Rode rename_columns antes.")
    
    allowed = {"realizado", "falta"}
    df["appointment_status"] = df["appointment_status"].astype("string").str.strip().str.lower()
    mask = df["appointment_status"].isin(allowed)
    return df[mask].copy()


def _convert_initial_types(df: pd.DataFrame) -> pd.DataFrame:
    """Tipagem inicial e limpeza."""
    df_clean = df.copy()
    for col in ["scheduled_at", "appointment_at"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce", dayfirst=True)

    if "patient_age" in df_clean.columns:
        df_clean["patient_age"] = pd.to_numeric(df_clean["patient_age"], errors="coerce")

    cat_cols = [
        "appointment_status", "patient_sex", "patient_city", "patient_neighborhood",
        "insurance_type", "unit_name", "unit_address", "unit_cep", "specialty"
    ]
    for c in cat_cols:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].astype("string").str.strip()
    return df_clean


def _create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Cria target binária: 1 = Falta (no-show), 0 = Realizado."""
    if "appointment_status" not in df.columns:
        raise KeyError("Coluna 'appointment_status' não encontrada.")
    
    status = df["appointment_status"].astype("string").str.strip().str.lower()
    df["no_show"] = np.where(status.eq("falta"), 1, 0).astype("int8")
    return df


def _create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features temporais: Lead Time, Sazonalidade, Ciclos e Períodos."""
    df_feat = df.copy()
    col_sched = 'scheduled_at'
    col_appt = 'appointment_at'

    if col_sched not in df_feat.columns or col_appt not in df_feat.columns:
        raise KeyError("Colunas de data não encontradas.")

    sched_normalized = df_feat[col_sched].dt.normalize()
    appt_normalized = df_feat[col_appt].dt.normalize()

    df_feat["waiting_days"] = (appt_normalized - sched_normalized).dt.days.clip(lower=0).astype(int)
    df_feat["is_same_day"] = (df_feat["waiting_days"] == 0).astype(int)
    df_feat["appointment_weekday"] = df_feat[col_appt].dt.weekday
    df_feat["is_weekend"] = df_feat["appointment_weekday"].isin([5, 6]).astype(int)
    df_feat["appointment_day_of_month"] = df_feat[col_appt].dt.day
    df_feat["appointment_week_of_month"] = df_feat[col_appt].dt.day.apply(lambda d: (d-1) // 7 + 1)
    df_feat["hour_appointment"] = df_feat[col_appt].dt.hour.astype('int8')

    bins = [-1, 6, 11, 14, 18, 24]
    labels = ['NIGHT', 'MORNING', 'MIDDAY', 'AFTERNOON', 'EVENING']
    df_feat['time_of_day'] = pd.cut(df_feat["hour_appointment"], bins=bins, labels=labels, right=False).astype('category')

    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat[col_appt].dt.month / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat[col_appt].dt.month / 12)
    df_feat['weekday_sin'] = np.sin(2 * np.pi * df_feat["appointment_weekday"] / 7)
    df_feat['weekday_cos'] = np.cos(2 * np.pi * df_feat["appointment_weekday"] / 7)
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat["hour_appointment"] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat["hour_appointment"] / 24)

    return df_feat


def _create_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula taxas históricas do paciente com proteção contra Data Leakage."""
    df = df.sort_values(by=['patient_id', 'appointment_at'])
    df["previous_appt_count"] = df.groupby('patient_id').cumcount()
    cumsum_target = df.groupby('patient_id')['no_show'].cumsum()
    df["no_show_rate_patient"] = ((cumsum_target - df['no_show']) / df["previous_appt_count"]).fillna(-1)
    return df


def _create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Discretiza a idade em faixas etárias (Binning)."""
    if "patient_age" in df.columns:
        df.loc[df["patient_age"] < 0, "patient_age"] = 0
        bins = [0, 2, 13, 18, 60, 120]
        labels = ["BABY", "CHILD", "TEEN", "ADULT", "SENIOR"]
        df["age_group"] = pd.cut(df["patient_age"], bins=bins, labels=labels, right=False).astype('category')
    return df


def _create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula recência e outras features comportamentais."""
    df_feat = df.copy()
    if 'patient_id' in df_feat.columns and 'appointment_at' in df_feat.columns:
        df_feat = df_feat.sort_values(by=['patient_id', 'appointment_at'])
        df_feat['last_appointment_date'] = df_feat.groupby('patient_id')['appointment_at'].shift(1)
        df_feat['days_since_last_visit'] = (df_feat['appointment_at'] - df_feat['last_appointment_date']).dt.days.fillna(-1).astype(int)
        df_feat = df_feat.drop(columns=['last_appointment_date'])

    if 'scheduled_at' in df_feat.columns and 'appointment_id' in df_feat.columns:
        batch_counts = df_feat.groupby(['patient_id', 'scheduled_at'])['appointment_id'].transform('count')
        df_feat['appointments_in_same_schedule'] = batch_counts.astype('int16')

    if 'patient_sex' in df_feat.columns and 'age_group' in df_feat.columns:
        df_feat['gender_age_profile'] = (df_feat['patient_sex'].astype(str) + "_" + df_feat['age_group'].astype(str)).astype('category')

    return df_feat


def _create_contextual_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Taxas históricas de No-Show por unidade e especialidade."""
    df_feat = df.copy()
    sort_cols = ["appointment_at"]
    if "appointment_id" in df_feat.columns: sort_cols.append("appointment_id")
    elif "scheduled_at" in df_feat.columns: sort_cols.append("scheduled_at")
    df_feat = df_feat.sort_values(sort_cols)

    for col in ["unit_name", "specialty"]:
        if col in df_feat.columns:
            df_feat[f"{col}_no_show_rate"] = df_feat.groupby(col)["no_show"].transform(lambda x: x.shift(1).expanding().mean())
    return df_feat


def _normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza textos para maiúsculas e remove espaços."""
    text_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df.loc[df[col] == 'NAN', col] = None
    return df


def create_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Cria 'is_holiday' (Brasil) baseado na data do appointment_at."""
    if "appointment_at" not in df.columns:
        raise KeyError("Coluna 'appointment_at' não encontrada.")
    try:
        import holidays
    except ImportError as e:
        raise ImportError("Pacote 'holidays' não instalado. Rode: pip install holidays") from e

    br_holidays = holidays.Brazil()
    df["is_holiday"] = df["appointment_at"].dt.date.isin(br_holidays).astype("int8")
    return df


def _reorder_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Reorganiza as colunas com base na ordem definida no YAML."""
    # Busca a lista no YAML; se não existir, mantém a ordem atual
    logical_order = config.get("column_order", [])

    if not logical_order:
        return df

    # Filtra apenas as colunas que realmente existem no DataFrame final
    existing_cols = [c for c in logical_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]

    return df[existing_cols + remaining_cols]