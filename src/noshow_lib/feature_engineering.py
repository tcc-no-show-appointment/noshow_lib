import pandas as pd
import numpy as np
import holidays
import re
from typing import Dict
from .logger import setup_logger

logger = setup_logger("noshow_lib.feature_engineering")

TARGET_COLUMN = "no_show"


def build_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Orquestra as transformações de feature engineering usando .pipe().

    Args:
        df: DataFrame original carregado do banco ou CSV.
        config: Dicionário de configuração (YAML).

    Returns:
        pd.DataFrame: DataFrame com todas as features calculadas (~59 features).
    """
    logger.info("Iniciando Pipeline de Feature Engineering...")
    initial_shape = df.shape

    df_processed = (
        df.copy()
        .pipe(_rename_columns, config)
        .pipe(_filter_noshow_status)
        .pipe(_convert_initial_types)
        .pipe(_create_cancellation_flag)
        .pipe(_create_target_variable)
        .pipe(_create_temporal_features)
        .pipe(_create_patient_history)
        .pipe(_create_age_groups)
        .pipe(_create_specialty_group)
        .pipe(_create_behavioral_features)
        .pipe(_create_contextual_rates)
        .pipe(_create_interaction_features)
        .pipe(_create_holiday_features)
        .pipe(_create_geo_features)
        .pipe(_normalize_text)
        .pipe(_reorder_columns, config)
    )

    logger.info(f"Feature engineering finalizado. Shape inicial: {initial_shape} -> Final: {df_processed.shape}")
    return df_processed


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 1 — Renomear colunas
# ─────────────────────────────────────────────────────────────────────────────
def _rename_columns(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Aplica o mapeamento do YAML e normaliza para snake_case."""
    column_map = config.get("column_map", {})
    if column_map:
        df = df.rename(columns=column_map)
    df.columns = [re.sub(r'[^0-9a-zA-Z_]+', '_', c).lower().strip('_') for c in df.columns]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 2 — Filtro de status
# ─────────────────────────────────────────────────────────────────────────────
def _filter_noshow_status(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas registros com desfecho binário conhecido (Realizado ou Falta)."""
    col = "appointment_status"
    if col not in df.columns:
        return df

    allowed = {"realizado", "falta"}
    status_norm = df[col].astype("string").str.strip().str.lower().fillna("")
    mask = status_norm.isin(allowed) | (status_norm == "") | status_norm.isna()
    filtered = df[mask].copy()
    logger.info(f"Filtragem de status: {len(df)} -> {len(filtered)} registros.")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 3 — Conversão de tipos
# ─────────────────────────────────────────────────────────────────────────────
def _convert_initial_types(df: pd.DataFrame) -> pd.DataFrame:
    """Tipagem inicial e limpeza básica de dados."""
    df = df.copy()

    for col in ["scheduled_at", "appointment_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "patient_age" in df.columns:
        df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce").fillna(0).astype("int16")

    cat_cols = [
        "appointment_status", "patient_sex", "patient_city", "patient_neighborhood",
        "insurance_type", "unit_name", "unit_address", "unit_cep", "specialty"
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 4 — Flag de cancelamento (temporária)
# ─────────────────────────────────────────────────────────────────────────────
def _create_cancellation_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Cria flag temporária de cancelamento para uso no histórico do paciente."""
    col = "appointment_status"
    if col not in df.columns:
        return df
    df = df.copy()
    s = df[col].astype("string").str.strip().str.lower()
    df["is_canceled_temp"] = s.str.contains("cancel", regex=False, na=False).astype("int8")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 5 — Variável alvo
# ─────────────────────────────────────────────────────────────────────────────
def _create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a variável alvo binária: 1 = Falta (no-show), 0 = Realizado."""
    col = "appointment_status"
    if col not in df.columns:
        return df

    s = df[col].astype("string").str.strip().str.lower()
    conditions = [
        (s == "falta").fillna(False).to_numpy(dtype=bool),
        (s == "realizado").fillna(False).to_numpy(dtype=bool),
    ]
    df[TARGET_COLUMN] = np.select(conditions, [1, 0], default=np.nan)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 6 — Features temporais
# ─────────────────────────────────────────────────────────────────────────────
def _create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features baseadas em tempo (Lead Time, Sazonalidade, Ciclos)."""
    df = df.copy()
    col_sched = "scheduled_at"
    col_appt = "appointment_at"

    if col_sched not in df.columns or col_appt not in df.columns:
        logger.error("Colunas de data (scheduled_at/appointment_at) ausentes.")
        return df

    sched_norm = df[col_sched].dt.normalize()
    appt_norm = df[col_appt].dt.normalize()
    df["waiting_days"] = (appt_norm - sched_norm).dt.days.clip(lower=0).astype("int16")
    df["is_same_day"] = (df["waiting_days"] == 0).astype("int8")

    appt = df[col_appt]
    df["appointment_weekday"] = appt.dt.weekday.astype("int8")
    df["is_weekend"] = appt.dt.weekday.isin([5, 6]).astype("int8")
    df["appointment_day_of_month"] = appt.dt.day.astype("int8")
    df["appointment_week_of_month"] = ((appt.dt.day - 1) // 7 + 1).astype("int8")
    df["is_month_start"] = (appt.dt.day <= 5).astype("int8")
    df["is_month_end"] = (appt.dt.day >= 25).astype("int8")
    df["hour_appointment"] = appt.dt.hour.fillna(0).astype("int8")

    bins = [-1, 6, 11, 14, 18, 24]
    labels = ["NIGHT", "MORNING", "MIDDAY", "AFTERNOON", "EVENING"]
    df["time_of_day"] = pd.cut(df["hour_appointment"], bins=bins, labels=labels, right=False).astype("category")

    df["month_sin"] = np.sin(2 * np.pi * appt.dt.month / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * appt.dt.month / 12).astype("float32")
    df["weekday_sin"] = np.sin(2 * np.pi * df["appointment_weekday"] / 7).astype("float32")
    df["weekday_cos"] = np.cos(2 * np.pi * df["appointment_weekday"] / 7).astype("float32")
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_appointment"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_appointment"] / 24).astype("float32")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 7 — Histórico do paciente (anti-leakage via shift)
# ─────────────────────────────────────────────────────────────────────────────
def _create_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula histórico completo do paciente com anti-leakage via shift(1)."""
    id_col = "patient_id"
    date_col = "appointment_at"

    if id_col not in df.columns or date_col not in df.columns:
        return df

    sort_cols = [id_col, date_col]
    if "appointment_id" in df.columns:
        sort_cols.append("appointment_id")
    df = df.sort_values(sort_cols).copy()

    grp = df.groupby(id_col, sort=False)

    # Contagem e antiguidade
    df["previous_appointments_count"] = grp.cumcount().astype("int32")
    df["has_patient_history"] = (df["previous_appointments_count"] > 0).astype("int8")
    df["patient_tenure_days"] = (
        df[date_col] - grp[date_col].transform("min")
    ).dt.days.fillna(0).astype("int32")

    # Cancelamentos (depende da flag temporária)
    if "is_canceled_temp" in df.columns:
        df["past_cancellations_count"] = (
            grp["is_canceled_temp"]
            .transform(lambda x: x.shift(1).cumsum().fillna(0))
            .astype("int32")
        )
        df["cancellation_rate"] = np.where(
            df["previous_appointments_count"] > 0,
            df["past_cancellations_count"] / df["previous_appointments_count"],
            0.0,
        ).astype("float32")
    else:
        df["past_cancellations_count"] = 0
        df["cancellation_rate"] = 0.0

    # Histórico de no-show (anti-leakage)
    if TARGET_COLUMN in df.columns:
        target_filled = df[TARGET_COLUMN].fillna(0)
        cumsum = target_filled.groupby(df[id_col]).cumsum()
        past_no_shows = (cumsum - target_filled).clip(lower=0).astype("int32")
        df["past_no_shows"] = past_no_shows

        prev = grp[TARGET_COLUMN].shift(1)
        df["previous_no_show"] = np.where(
            df["previous_appointments_count"] == 0, -1, prev.fillna(-1)
        ).astype("int8")

        prev2 = grp[TARGET_COLUMN].shift(2)
        df["consecutive_no_shows_2"] = np.where(
            df["previous_appointments_count"] < 2,
            -1,
            ((prev.fillna(0) == 1) & (prev2.fillna(0) == 1)).astype("int8"),
        ).astype("int8")

        df["no_show_rate_patient_smoothed"] = (
            (past_no_shows + 1) / (df["previous_appointments_count"] + 2)
        ).astype("float32")
        df["no_show_rate_patient"] = np.where(
            df["previous_appointments_count"] == 0,
            -1.0,
            df["no_show_rate_patient_smoothed"],
        ).astype("float32")

        rate3 = grp[TARGET_COLUMN].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
        rate5 = grp[TARGET_COLUMN].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        df["no_show_rate_recent_3"] = np.where(
            df["previous_appointments_count"] == 0, -1.0, rate3
        ).astype("float32")
        df["no_show_rate_recent_5"] = np.where(
            df["previous_appointments_count"] == 0, -1.0, rate5
        ).astype("float32")

        # Dias desde o último no-show
        df["_ns_date_tmp"] = df[date_col].where(df[TARGET_COLUMN] == 1)
        last_ns = grp["_ns_date_tmp"].transform(lambda x: x.shift(1).ffill())
        df["days_since_last_no_show"] = np.where(
            last_ns.isna(), -1, (df[date_col] - last_ns).dt.days
        ).astype("float32")
        df.drop(columns=["_ns_date_tmp"], inplace=True)
    else:
        for col in [
            "past_no_shows", "previous_no_show", "consecutive_no_shows_2",
            "no_show_rate_patient", "no_show_rate_patient_smoothed",
            "no_show_rate_recent_3", "no_show_rate_recent_5", "days_since_last_no_show",
        ]:
            df[col] = -1.0

    # Dias desde a última consulta
    df["_last_visit_tmp"] = grp[date_col].shift(1)
    df["days_since_last_visit"] = (
        (df[date_col] - df["_last_visit_tmp"]).dt.days.fillna(-1).astype("float32")
    )
    df.drop(columns=["_last_visit_tmp"], inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 8 — Grupos etários
# ─────────────────────────────────────────────────────────────────────────────
def _create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Categoriza a idade em grupos etários."""
    col = "patient_age"
    if col not in df.columns:
        return df

    df = df.copy()
    df.loc[df[col] < 0, col] = 0

    bins = [-1, 2, 12, 17, 59, 200]
    labels = ["BABY", "CHILD", "TEEN", "ADULT", "SENIOR"]
    df["age_group"] = pd.cut(df[col], bins=bins, labels=labels).astype("category")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 9 — Grupo de especialidade
# ─────────────────────────────────────────────────────────────────────────────
def _create_specialty_group(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa especialidades em categorias clínicas."""
    col = "specialty"
    if col not in df.columns:
        return df

    s = df[col].astype(str).str.upper()
    conditions = [
        s.str.contains(r"ORTOPE", na=False),
        s.str.contains(r"CIRU|ANGI|VASCU|TORCIC", na=False),
        s.str.contains(r"PSIQUI|PSICOL|DEPEND|MENTAL", na=False),
        s.str.contains(r"FISIO|FONOAUDIO|TERAPIA OCUPACIONAL|FISIATRIA", na=False),
        s.str.contains(r"GINECO|OBSTETR|PRE-NATAL|PEDIAT|INFANTIL", na=False),
        s.str.contains(r"CLINICA GERAL|GERONTOL|HOMEOPAT|TELEMEDICINA|FAMILIA|TRIAGEM|ENTREVISTA|PLANTAO", na=False),
        s.str.contains(r"EXAMES|QUIMIO|RETINA|ANUSCOPIA|ACOMPANHAMENTO", na=False),
        s.str.contains(r"CARDIO|DERMATO|ENDOCRINO|GASTRO|NEURO|OFTALMO|ONCO|REUMATO|PNEUMO|HEMATO|INFECTO|ALERGO|OTORRINO|NUTRI|HEPATO|GLAUCOMA", na=False),
    ]
    choices = [
        "ORTOPEDIA", "CIRURGICO_E_VASCULAR", "SAUDE_MENTAL", "TERAPIAS_REABILITACAO",
        "MATERNO_INFANTIL", "CLINICA_GERAL_E_TRIAGEM", "EXAMES_E_PROCEDIMENTOS", "CLINICA_ESPECIALIZADA",
    ]
    df["specialty_group"] = np.select(conditions, choices, default="OUTRAS_ESPECIALIDADES")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 10 — Features comportamentais
# ─────────────────────────────────────────────────────────────────────────────
def _create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de recência, variação e agendamento em lote."""
    df = df.copy()
    id_col = "patient_id"
    date_col = "appointment_at"

    if id_col not in df.columns or date_col not in df.columns:
        return df

    df = df.sort_values(by=[id_col, date_col])
    grp = df.groupby(id_col, sort=False)

    # Diferença de waiting_days entre consultas do mesmo paciente (variação de comportamento)
    if "waiting_days" in df.columns:
        prev_waiting = grp["waiting_days"].shift(1)
        df["waiting_days_delta"] = (df["waiting_days"] - prev_waiting).fillna(0).astype("float32")
    else:
        df["waiting_days_delta"] = 0.0

    # Mudança de especialidade em relação à consulta anterior
    if "specialty" in df.columns:
        prev_specialty = grp["specialty"].shift(1)
        df["is_diff_specialty"] = (
            (df["specialty"] != prev_specialty) & prev_specialty.notna()
        ).astype("int8")
    else:
        df["is_diff_specialty"] = 0

    # Consultas agendadas no mesmo momento (batching)
    if "scheduled_at" in df.columns:
        df["appointments_in_same_schedule_day"] = (
            df.groupby([id_col, "scheduled_at"])["appointment_at"]
            .transform("count")
            .astype("int16")
        )
    else:
        df["appointments_in_same_schedule_day"] = 1

    # Perfil demográfico combinado
    if "patient_sex" in df.columns and "age_group" in df.columns:
        df["gender_age_profile"] = (
            df["patient_sex"].astype(str) + "_" + df["age_group"].astype(str)
        ).astype("category")
    else:
        df["gender_age_profile"] = "UNKNOWN"

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 11 — Taxas contextuais (anti-leakage via expanding mean)
# ─────────────────────────────────────────────────────────────────────────────
def _create_contextual_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Taxas de No-Show por contexto usando expanding mean com anti-leakage."""
    df = df.copy()
    if "appointment_at" not in df.columns:
        return df

    sort_cols = ["appointment_at"]
    if "appointment_id" in df.columns:
        sort_cols.append("appointment_id")
    df = df.sort_values(sort_cols)

    has_target = TARGET_COLUMN in df.columns

    rate_map = {
        "unit_name": "unit_no_show_rate",
        "specialty": "specialty_no_show_rate",
        "specialty_group": "specialty_group_no_show_rate",
        "patient_neighborhood": "neighborhood_risk_score",
        "insurance_type": "insurance_no_show_rate",
    }

    for group_col, out_col in rate_map.items():
        if group_col not in df.columns:
            continue
        if has_target:
            df[out_col] = (
                df.groupby(group_col, observed=True)[TARGET_COLUMN]
                .transform(lambda x: x.shift(1).expanding().mean())
                .astype("float32")
            )
        else:
            df[out_col] = -1.0

    # Flag de especialidade de alto risco
    if "specialty_group_no_show_rate" in df.columns:
        threshold = np.float32(0.35)
        df["specialty_high_no_show_flag"] = np.where(
            df["specialty_group_no_show_rate"].isna(),
            0,
            (df["specialty_group_no_show_rate"] >= threshold).astype("int8"),
        ).astype("int8")
    else:
        df["specialty_high_no_show_flag"] = 0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 12 — Features de interação
# ─────────────────────────────────────────────────────────────────────────────
def _create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de interação entre variáveis numéricas."""
    df = df.copy()

    if "patient_age" in df.columns and "waiting_days" in df.columns:
        df["age_x_waiting_days"] = (
            df["patient_age"].astype("float32") * df["waiting_days"].astype("float32")
        ).astype("float32")
    else:
        df["age_x_waiting_days"] = 0.0

    if "no_show_rate_patient" in df.columns and "waiting_days" in df.columns:
        rate = df["no_show_rate_patient"].clip(lower=0)
        df["no_show_rate_x_waiting_days"] = (
            rate * df["waiting_days"].astype("float32")
        ).astype("float32")
    else:
        df["no_show_rate_x_waiting_days"] = 0.0

    if "patient_age" in df.columns and "specialty_group_no_show_rate" in df.columns:
        rate = df["specialty_group_no_show_rate"].clip(lower=0)
        df["age_x_specialty_risk"] = (
            df["patient_age"].astype("float32") * rate
        ).astype("float32")
    else:
        df["age_x_specialty_risk"] = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 13 — Feriados
# ─────────────────────────────────────────────────────────────────────────────
def _create_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Identifica feriados e janelas próximas (pré, pós, ponte)."""
    col = "appointment_at"
    if col not in df.columns:
        return df

    df = df.copy()
    br_holidays = holidays.Brazil()
    dates = df[col].dt.date

    df["is_holiday"] = dates.isin(br_holidays).astype("int8")

    df["_date_tmp"] = pd.to_datetime(dates)
    df["is_pre_holiday"] = (df["_date_tmp"] + pd.Timedelta(days=1)).dt.date.isin(br_holidays).astype("int8")
    df["is_post_holiday"] = (df["_date_tmp"] - pd.Timedelta(days=1)).dt.date.isin(br_holidays).astype("int8")

    # Ponte: segunda após feriado na sexta, ou sexta antes de feriado na segunda
    df["is_bridge_day"] = (
        ((df["appointment_weekday"] == 4) & df["is_pre_holiday"].astype(bool)) |
        ((df["appointment_weekday"] == 0) & df["is_post_holiday"].astype(bool))
    ).astype("int8")

    df["is_holiday_window"] = (
        df[["is_holiday", "is_pre_holiday", "is_post_holiday", "is_bridge_day"]].max(axis=1)
    ).astype("int8")

    df.drop(columns=["_date_tmp"], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 14 — Features geográficas
# ─────────────────────────────────────────────────────────────────────────────
def _create_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula proximidade geográfica paciente-unidade."""
    df = df.copy()

    # same_cep_prefix5: compara CEP do paciente (se disponível) com CEP da unidade
    if "patient_cep" in df.columns and "unit_cep" in df.columns:
        pcep = df["patient_cep"].astype("string").str.replace(r"\D", "", regex=True)
        ucep = df["unit_cep"].astype("string").str.replace(r"\D", "", regex=True)
        valid = pcep.str.len().ge(5) & ucep.str.len().ge(5)
        df["same_cep_prefix5"] = np.where(
            valid, pcep.str[:5].eq(ucep.str[:5]).fillna(False).astype("int8"), -1
        ).astype("int8")
    else:
        df["same_cep_prefix5"] = -1

    # same_city: cidade do paciente vs cidade extraída do endereço da unidade
    pcity = (
        df["patient_city"].astype("string").str.upper().str.strip()
        if "patient_city" in df.columns
        else pd.Series("", index=df.index, dtype="string")
    )
    ucity = (
        df["unit_address"].astype("string").str.upper()
        .str.extract(r"-\s*([A-ZÀ-Ú\s]+)$", expand=False)
        .str.strip()
        if "unit_address" in df.columns
        else pd.Series("", index=df.index, dtype="string")
    )
    df["same_city"] = (pcity == ucity).fillna(False).astype("int8")

    # is_local_resident: paciente na mesma cidade ou bairro da unidade
    df["is_local_resident"] = df["same_city"].copy()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 15 — Normalização de texto
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas de texto (uppercase, sem espaços extras)."""
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df.loc[df[col] == "NAN", col] = None
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Etapa 16 — Reordenar colunas
# ─────────────────────────────────────────────────────────────────────────────
def _reorder_columns(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Organiza as colunas conforme definido no YAML."""
    logical_order = config.get("column_order", [])
    if not logical_order:
        return df

    existing = [c for c in logical_order if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df[existing + remaining]
