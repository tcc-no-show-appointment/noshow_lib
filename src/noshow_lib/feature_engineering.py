import pandas as pd
import numpy as np
import holidays
import re
from typing import Dict, List, Optional
from .logger import setup_logger

logger = setup_logger("noshow_lib.feature_engineering")

def build_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Orquestra as transformações de feature engineering usando .pipe().
    
    Args:
        df: DataFrame original carregado do banco.
        config: Dicionário de configuração (YAML).
        
    Returns:
        pd.DataFrame: DataFrame com todas as features calculadas.
    """
    logger.info("Iniciando Pipeline de Feature Engineering...")
    initial_shape = df.shape

    df_processed = (
        df.copy()
        .pipe(_rename_columns, config)
        .pipe(_filter_noshow_status)
        .pipe(_convert_initial_types)
        .pipe(_create_target_variable)
        .pipe(_create_temporal_features)
        .pipe(_create_patient_history)
        .pipe(_create_age_groups)
        .pipe(_create_behavioral_features)
        .pipe(_create_contextual_rates)
        .pipe(_create_holiday_feature)
        .pipe(_normalize_text)
        .pipe(_reorder_columns, config)
    )

    logger.info(f"Feature engineering finalizado. Shape inicial: {initial_shape} -> Final: {df_processed.shape}")
    return df_processed

def _rename_columns(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Aplica o mapeamento do YAML e normaliza para snake_case."""
    column_map = config.get("column_map", {})
    if column_map:
        df = df.rename(columns=column_map)
    
    # Normalização para snake_case (remove caracteres especiais e espaços)
    df.columns = [re.sub(r'[^0-9a-zA-Z_]+', '_', c).lower().strip('_') for c in df.columns]
    logger.debug(f"Colunas normalizadas: {df.columns.tolist()[:5]}...")
    return df

def _filter_noshow_status(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas registros com desfecho binário conhecido (Realizado ou Falta)."""
    col = "appointment_status"
    if col not in df.columns:
        logger.warning(f"Coluna '{col}' não encontrada para filtragem de status.")
        return df
    
    allowed = {"realizado", "falta"}
    df[col] = df[col].astype("string").str.strip().str.lower()
    mask = df[col].isin(allowed)
    filtered_df = df[mask].copy()
    
    logger.info(f"Filtragem de status: {len(df)} -> {len(filtered_df)} registros.")
    return filtered_df

def _convert_initial_types(df: pd.DataFrame) -> pd.DataFrame:
    """Tipagem inicial e limpeza básica de dados."""
    df_clean = df.copy()
    
    # Datas
    for col in ["scheduled_at", "appointment_at"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors="coerce")
    
    # Numéricos
    if "patient_age" in df_clean.columns:
        df_clean["patient_age"] = pd.to_numeric(df_clean["patient_age"], errors="coerce").fillna(0)

    # Categóricos
    cat_cols = [
        "appointment_status", "patient_sex", "patient_city", "patient_neighborhood",
        "insurance_type", "unit_name", "unit_address", "unit_cep", "specialty"
    ]
    for c in cat_cols:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].astype("string").str.strip()
            
    return df_clean

def _create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a variável alvo binária: 1 = Falta (no-show), 0 = Realizado."""
    col = "appointment_status"
    if col not in df.columns:
        return df
    
    df["no_show"] = np.where(df[col].str.lower() == "falta", 1, 0).astype("int8")
    return df

def _create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features baseadas em tempo (Lead Time, Sazonalidade, Ciclos)."""
    df_feat = df.copy()
    col_sched = 'scheduled_at'
    col_appt = 'appointment_at'

    if col_sched not in df_feat.columns or col_appt not in df_feat.columns:
        logger.error("Colunas de data (scheduled_at/appointment_at) ausentes.")
        return df_feat

    # Lead Time (Dias de espera)
    sched_normalized = df_feat[col_sched].dt.normalize()
    appt_normalized = df_feat[col_appt].dt.normalize()
    df_feat["waiting_days"] = (appt_normalized - sched_normalized).dt.days.clip(lower=0).astype(int)
    df_feat["is_same_day"] = (df_feat["waiting_days"] == 0).astype(int)

    # Componentes de Data
    df_feat["appointment_weekday"] = df_feat[col_appt].dt.weekday
    df_feat["is_weekend"] = df_feat["appointment_weekday"].isin([5, 6]).astype(int)
    df_feat["appointment_day_of_month"] = df_feat[col_appt].dt.day
    df_feat["appointment_week_of_month"] = (df_feat[col_appt].dt.day - 1) // 7 + 1
    df_feat["hour_appointment"] = df_feat[col_appt].dt.hour.fillna(0).astype('int8')

    # Períodos do dia
    bins = [-1, 6, 11, 14, 18, 24]
    labels = ['NIGHT', 'MORNING', 'MIDDAY', 'AFTERNOON', 'EVENING']
    df_feat['time_of_day'] = pd.cut(df_feat["hour_appointment"], bins=bins, labels=labels, right=False).astype('category')

    # Codificação Cíclica (Seno/Cosseno)
    df_feat['month_sin'] = np.sin(2 * np.pi * df_feat[col_appt].dt.month / 12)
    df_feat['month_cos'] = np.cos(2 * np.pi * df_feat[col_appt].dt.month / 12)
    df_feat['weekday_sin'] = np.sin(2 * np.pi * df_feat["appointment_weekday"] / 7)
    df_feat['weekday_cos'] = np.cos(2 * np.pi * df_feat["appointment_weekday"] / 7)
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat["hour_appointment"] / 24)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat["hour_appointment"] / 24)

    return df_feat

def _create_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula histórico do paciente (Taxa de No-Show acumulada)."""
    id_col = 'patient_id'
    date_col = 'appointment_at'
    
    if id_col not in df.columns or date_col not in df.columns or 'no_show' not in df.columns:
        return df

    df = df.sort_values(by=[id_col, date_col])
    
    # Contagem de consultas anteriores
    df["previous_appointments_count"] = df.groupby(id_col).cumcount()
    
    # Taxa de No-Show (excluindo a consulta atual para evitar leakage)
    cumsum_target = df.groupby(id_col)['no_show'].cumsum()
    df["no_show_rate_patient"] = ((cumsum_target - df['no_show']) / df["previous_appointments_count"]).fillna(-1)
    
    return df

def _create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Categoriza a idade em grupos etários."""
    col = "patient_age"
    if col not in df.columns:
        return df

    df.loc[df[col] < 0, col] = 0
    bins = [0, 2, 13, 18, 60, 150]
    labels = ["BABY", "CHILD", "TEEN", "ADULT", "SENIOR"]
    df["age_group"] = pd.cut(df[col], bins=bins, labels=labels, right=False).astype('category')
    return df

def _create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de comportamento (recência e agendamentos em lote)."""
    df_feat = df.copy()
    id_col = 'patient_id'
    date_col = 'appointment_at'

    # Recência (Dias desde a última consulta)
    if id_col in df_feat.columns and date_col in df_feat.columns:
        df_feat = df_feat.sort_values(by=[id_col, date_col])
        df_feat['last_visit'] = df_feat.groupby(id_col)[date_col].shift(1)
        df_feat['days_since_last_visit'] = (df_feat[date_col] - df_feat['last_visit']).dt.days.fillna(-1).astype(int)
        df_feat = df_feat.drop(columns=['last_visit'])

    # Consultas agendadas no mesmo momento (Batching)
    if id_col in df_feat.columns and 'scheduled_at' in df_feat.columns:
        df_feat['appointments_in_same_schedule'] = df_feat.groupby([id_col, 'scheduled_at'])['appointment_at'].transform('count').astype('int16')

    return df_feat

def _create_contextual_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Taxas de No-Show por contexto (Especialidade e Unidade)."""
    df_feat = df.copy()
    if 'no_show' not in df_feat.columns or 'appointment_at' not in df_feat.columns:
        return df_feat

    df_feat = df_feat.sort_values('appointment_at')
    
    for col in ["unit_name", "specialty"]:
        if col in df_feat.columns:
            # Nome amigável para o config (unit_name -> unit_no_show_rate)
            feature_name = f"{col.replace('_name', '')}_no_show_rate"
            
            # Média expansiva excluindo o registro atual (leakage protection)
            df_feat[feature_name] = df_feat.groupby(col)["no_show"].transform(
                lambda x: x.shift(1).expanding().mean()
            ).fillna(-1)
            
    return df_feat

def _create_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Identifica feriados nacionais."""
    col = "appointment_at"
    if col not in df.columns:
        return df
        
    br_holidays = holidays.Brazil()
    df["is_holiday"] = df[col].dt.date.isin(br_holidays).astype("int8")
    return df

def _normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas de texto (uppercase, sem espaços extras)."""
    text_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df.loc[df[col] == 'NAN', col] = None
    return df

def _reorder_columns(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Organiza as colunas conforme definido no YAML."""
    logical_order = config.get("column_order", [])
    if not logical_order:
        return df

    existing_cols = [c for c in logical_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    
    return df[existing_cols + remaining_cols]
