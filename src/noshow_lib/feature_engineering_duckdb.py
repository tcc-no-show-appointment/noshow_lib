"""Feature engineering via DuckDB for large Parquet datasets.

Parquet in → Parquet out, sem carregar em memória.
Executa as mesmas 16 etapas de feature_engineering.py via SQL columnar.

NOTA (0.4.0): o pipeline pandas (`feature_engineering.py`) foi enxugado para
produzir apenas as 32 features consumidas pelos 8 modelos LightGBM + a feature
nova `cluster_patient` (K-Means). Este módulo DuckDB ainda gera as 59 features
originais via SQL (poda equivalente em SQL fica como trabalho futuro). Os
consumidores devem filtrar a saída para as 32 features esperadas se forem usar
o resultado com modelos treinados na 0.4.0. A feature `cluster_patient` NÃO é
gerada por este módulo — precisa ser aplicada via pandas após o DuckDB:

    output = build_features_from_parquet("raw.parquet", config, "features.parquet")
    from noshow_lib.feature_engineering import apply_patient_cluster, load_cluster_artifact
    df = pd.read_parquet(output)
    df = apply_patient_cluster(df, load_cluster_artifact(Path("models/")))
"""

import holidays
import pandas as pd
import duckdb
from pathlib import Path
from typing import Dict, Optional, Union

from .logger import setup_logger

logger = setup_logger("noshow_lib.feature_engineering_duckdb")


def build_features_from_parquet(
    parquet_path: Union[str, Path],
    config: Dict,
    output_path: Union[str, Path],
    history_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Executa feature engineering via DuckDB em dados Parquet.

    Parquet in → Parquet out, sem carregar em memória.
    Traduz as mesmas 16 etapas do build_features() pandas para SQL columnar.

    Args:
        parquet_path: Caminho do Parquet de entrada (colunas raw ou internas).
        config: Dicionário de configuração (YAML).
        output_path: Caminho do Parquet de saída com as features calculadas.
        history_path: Opcional. Parquet com histórico do paciente para inferência
                      (anti-leakage). Deve conter desfechos conhecidos (no_show).

    Returns:
        str: output_path — para encadeamento com train_model/predict.

    Example::

        output = build_features_from_parquet("raw.parquet", config, "features.parquet")
        df = pd.read_parquet(output)
        train_model(df, config)
    """
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet de entrada não encontrado: {parquet_path}")

    if history_path is not None:
        history_path = Path(history_path)
        if not history_path.exists():
            raise FileNotFoundError(
                f"Parquet de histórico não encontrado: {history_path}"
            )

    logger.info(f"Iniciando Feature Engineering DuckDB: {parquet_path}")

    con = duckdb.connect()

    _register_holidays(con)

    parquet_cols = _get_parquet_columns(con, str(parquet_path))
    column_map = config.get("column_map", {})
    uses_raw = bool(set(parquet_cols) & set(column_map.keys()))
    effective_map = column_map if uses_raw else {}

    logger.info(
        f"Colunas detectadas: {len(parquet_cols)}. Usa nomes raw: {uses_raw}. "
        f"History: {'sim' if history_path else 'não'}."
    )

    sql = _build_full_sql(
        parquet_path=str(parquet_path),
        config=config,
        output_path=str(output_path),
        history_path=str(history_path) if history_path else None,
        column_map=effective_map,
        parquet_cols=parquet_cols,
    )

    logger.debug(f"SQL gerado ({len(sql)} chars). Executando...")
    con.execute(sql)

    logger.info(f"Feature engineering DuckDB concluído. Output: {output_path}")
    return str(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _get_parquet_columns(con: duckdb.DuckDBPyConnection, path: str) -> list:
    """Retorna lista de colunas do Parquet sem carregar os dados."""
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{_esc(path)}')"
    ).fetchall()
    return [row[0] for row in rows]


def _register_holidays(
    con: duckdb.DuckDBPyConnection, years: tuple = (2015, 2036)
) -> None:
    """Registra feriados brasileiros como tabela DuckDB temporária."""
    br = holidays.Brazil(years=range(*years))
    # Manter como datetime.date (não TIMESTAMP) para que o JOIN com DATE funcione corretamente
    hol_df = pd.DataFrame({"holiday_date": list(br.keys())})
    con.register("br_holidays", hol_df)


def _esc(s: str) -> str:
    """Escapa um caminho para uso em string literals SQL."""
    return s.replace("'", "''").replace("\\", "/")


def _build_input_selects(column_map: dict, parquet_cols: list) -> str:
    """
    Constrói o SELECT da CTE de entrada, renomeando colunas raw para internas.
    Colunas ausentes são retornadas como NULL.
    """
    reverse_map = {v: k for k, v in column_map.items()}

    # Colunas internas necessárias e seus nomes raw correspondentes
    needed = [
        "appointment_id",
        "patient_id",
        "appointment_status",
        "scheduled_at",
        "appointment_at",
        "patient_age",
        "patient_sex",
        "patient_city",
        "patient_neighborhood",
        "insurance_type",
        "unit_name",
        "unit_address",
        "unit_cep",
        "specialty",
        "patient_cep",  # opcional — ausente em muitos datasets
    ]

    lines = []
    for internal in needed:
        raw = reverse_map.get(internal, internal)
        if raw in parquet_cols:
            lines.append(f'        "{raw}" AS {internal}' if raw != internal else f'        "{internal}"')
        elif internal in parquet_cols:
            lines.append(f'        "{internal}"')
        else:
            lines.append(f"        NULL AS {internal}")

    return ",\n".join(lines)


def _build_final_select(column_order: list) -> str:
    """Constrói o SELECT final com as colunas na ordem definida no config."""
    if not column_order:
        return "*"
    return ",\n        ".join(column_order)


# ─────────────────────────────────────────────────────────────────────────────
# Builder SQL principal
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_sql(
    parquet_path: str,
    config: Dict,
    output_path: str,
    history_path: Optional[str],
    column_map: Dict,
    parquet_cols: list,
) -> str:
    """Monta o SQL completo com todas as CTEs encadeadas."""

    column_order = config.get("column_order", [])
    input_selects = _build_input_selects(column_map, parquet_cols)
    final_cols = _build_final_select(column_order)
    p_in = _esc(parquet_path)
    p_out = _esc(output_path)

    # ─── CTE 0: Input (+ history se fornecido) ───────────────────────────────
    input_cte = f"""input AS (
    SELECT
{input_selects},
        TRUE AS is_new_row
    FROM read_parquet('{p_in}')
)"""

    if history_path:
        p_hist = _esc(history_path)
        input_cte += f""",
history_input AS (
    SELECT
{input_selects},
        FALSE AS is_new_row
    FROM read_parquet('{p_hist}')
),
all_data AS (
    SELECT * FROM input
    UNION ALL
    SELECT * FROM history_input
)"""
    else:
        input_cte += """,
all_data AS (
    SELECT * FROM input
)"""

    # ─── CTE 1: Base — filter + types + cancellation flag + target ───────────
    base_cte = """base AS (
    SELECT
        CAST(appointment_id   AS VARCHAR) AS appointment_id,
        CAST(patient_id       AS VARCHAR) AS patient_id,
        TRIM(CAST(appointment_status AS VARCHAR)) AS appointment_status,
        TRY_CAST(scheduled_at  AS TIMESTAMP) AS scheduled_at,
        TRY_CAST(appointment_at AS TIMESTAMP) AS appointment_at,
        GREATEST(0, COALESCE(TRY_CAST(patient_age AS SMALLINT), 0)) AS patient_age,
        TRIM(CAST(patient_sex          AS VARCHAR)) AS patient_sex,
        TRIM(CAST(patient_city         AS VARCHAR)) AS patient_city,
        TRIM(CAST(patient_neighborhood AS VARCHAR)) AS patient_neighborhood,
        TRIM(CAST(insurance_type       AS VARCHAR)) AS insurance_type,
        TRIM(CAST(unit_name            AS VARCHAR)) AS unit_name,
        TRIM(CAST(unit_address         AS VARCHAR)) AS unit_address,
        TRIM(CAST(unit_cep             AS VARCHAR)) AS unit_cep,
        CAST(patient_cep               AS VARCHAR)  AS patient_cep,
        UPPER(TRIM(CAST(specialty      AS VARCHAR))) AS specialty,
        is_new_row,
        -- Etapa 4: flag temporária de cancelamento
        CASE WHEN LOWER(TRIM(CAST(appointment_status AS VARCHAR))) LIKE '%cancel%'
             THEN 1 ELSE 0 END AS is_canceled_temp,
        -- Etapa 5: variável alvo
        CASE
            WHEN LOWER(TRIM(CAST(appointment_status AS VARCHAR))) = 'falta'     THEN 1
            WHEN LOWER(TRIM(CAST(appointment_status AS VARCHAR))) = 'realizado' THEN 0
            ELSE NULL
        END AS no_show
    FROM all_data
    WHERE LOWER(TRIM(CAST(appointment_status AS VARCHAR))) IN ('realizado', 'falta')
       OR TRIM(CAST(appointment_status AS VARCHAR)) = ''
       OR appointment_status IS NULL
)"""

    # ─── CTE 2: Temporal features ─────────────────────────────────────────────
    temporal_cte = """temporal AS (
    SELECT *,
        -- Etapa 6: Lead time
        GREATEST(0, DATEDIFF('day',
            DATE_TRUNC('day', scheduled_at),
            DATE_TRUNC('day', appointment_at)
        ))::SMALLINT AS waiting_days,
        CASE WHEN GREATEST(0, DATEDIFF('day',
            DATE_TRUNC('day', scheduled_at),
            DATE_TRUNC('day', appointment_at)
        )) = 0 THEN 1 ELSE 0 END::TINYINT AS is_same_day,
        -- Weekday: DuckDB DOW (0=Sun) → pandas weekday (0=Mon)
        ((EXTRACT(DOW FROM appointment_at)::INT + 6) % 7)::TINYINT AS appointment_weekday,
        CASE WHEN EXTRACT(DOW FROM appointment_at) IN (0, 6) THEN 1 ELSE 0 END::TINYINT AS is_weekend,
        EXTRACT(DAY FROM appointment_at)::TINYINT AS appointment_day_of_month,
        -- Usar // (divisão inteira) porque INT / 7 em DuckDB retorna DOUBLE, e cast para TINYINT arredonda
        (((EXTRACT(DAY FROM appointment_at)::INT - 1) // 7) + 1)::TINYINT AS appointment_week_of_month,
        CASE WHEN EXTRACT(DAY FROM appointment_at) <= 5  THEN 1 ELSE 0 END::TINYINT AS is_month_start,
        CASE WHEN EXTRACT(DAY FROM appointment_at) >= 25 THEN 1 ELSE 0 END::TINYINT AS is_month_end,
        COALESCE(EXTRACT(HOUR FROM appointment_at), 0)::TINYINT AS hour_appointment,
        -- Período do dia (right=False: [left, right) → pd.cut com bins [-1,6,11,14,18,24])
        CASE
            WHEN COALESCE(EXTRACT(HOUR FROM appointment_at), 0) < 6  THEN 'NIGHT'
            WHEN COALESCE(EXTRACT(HOUR FROM appointment_at), 0) < 11 THEN 'MORNING'
            WHEN COALESCE(EXTRACT(HOUR FROM appointment_at), 0) < 14 THEN 'MIDDAY'
            WHEN COALESCE(EXTRACT(HOUR FROM appointment_at), 0) < 18 THEN 'AFTERNOON'
            ELSE 'EVENING'
        END AS time_of_day,
        -- Encodings cíclicos
        SIN(2 * pi() * EXTRACT(MONTH FROM appointment_at) / 12.0)::FLOAT AS month_sin,
        COS(2 * pi() * EXTRACT(MONTH FROM appointment_at) / 12.0)::FLOAT AS month_cos,
        SIN(2 * pi() * ((EXTRACT(DOW FROM appointment_at)::INT + 6) % 7) / 7.0)::FLOAT AS weekday_sin,
        COS(2 * pi() * ((EXTRACT(DOW FROM appointment_at)::INT + 6) % 7) / 7.0)::FLOAT AS weekday_cos,
        SIN(2 * pi() * COALESCE(EXTRACT(HOUR FROM appointment_at), 0) / 24.0)::FLOAT AS hour_sin,
        COS(2 * pi() * COALESCE(EXTRACT(HOUR FROM appointment_at), 0) / 24.0)::FLOAT AS hour_cos
    FROM base
)"""

    # ─── CTE 3: Grupos etários + grupo de especialidade ───────────────────────
    enriched_cte = """enriched AS (
    SELECT *,
        -- Etapa 8: grupos etários (bins (-1,2],(2,12],(12,17],(17,59],(59,200])
        CASE
            WHEN patient_age <= 2  THEN 'BABY'
            WHEN patient_age <= 12 THEN 'CHILD'
            WHEN patient_age <= 17 THEN 'TEEN'
            WHEN patient_age <= 59 THEN 'ADULT'
            ELSE 'SENIOR'
        END AS age_group,
        -- Etapa 9: grupo de especialidade
        CASE
            WHEN specialty LIKE '%ORTOPE%' THEN 'ORTOPEDIA'
            WHEN specialty LIKE '%CIRU%'   OR specialty LIKE '%ANGI%'
              OR specialty LIKE '%VASCU%'  OR specialty LIKE '%TORCIC%'
              THEN 'CIRURGICO_E_VASCULAR'
            WHEN specialty LIKE '%PSIQUI%'  OR specialty LIKE '%PSICOL%'
              OR specialty LIKE '%DEPEND%'  OR specialty LIKE '%MENTAL%'
              THEN 'SAUDE_MENTAL'
            WHEN specialty LIKE '%FISIO%'        OR specialty LIKE '%FONOAUDIO%'
              OR specialty LIKE '%TERAPIA OCUPACIONAL%' OR specialty LIKE '%FISIATRIA%'
              THEN 'TERAPIAS_REABILITACAO'
            WHEN specialty LIKE '%GINECO%'   OR specialty LIKE '%OBSTETR%'
              OR specialty LIKE '%PRE-NATAL%' OR specialty LIKE '%PEDIAT%'
              OR specialty LIKE '%INFANTIL%'  THEN 'MATERNO_INFANTIL'
            WHEN specialty LIKE '%CLINICA GERAL%' OR specialty LIKE '%GERONTOL%'
              OR specialty LIKE '%HOMEOPAT%'   OR specialty LIKE '%TELEMEDICINA%'
              OR specialty LIKE '%FAMILIA%'    OR specialty LIKE '%TRIAGEM%'
              OR specialty LIKE '%ENTREVISTA%' OR specialty LIKE '%PLANTAO%'
              THEN 'CLINICA_GERAL_E_TRIAGEM'
            WHEN specialty LIKE '%EXAMES%'       OR specialty LIKE '%QUIMIO%'
              OR specialty LIKE '%RETINA%'       OR specialty LIKE '%ANUSCOPIA%'
              OR specialty LIKE '%ACOMPANHAMENTO%' THEN 'EXAMES_E_PROCEDIMENTOS'
            WHEN specialty LIKE '%CARDIO%'   OR specialty LIKE '%DERMATO%'
              OR specialty LIKE '%ENDOCRINO%' OR specialty LIKE '%GASTRO%'
              OR specialty LIKE '%NEURO%'    OR specialty LIKE '%OFTALMO%'
              OR specialty LIKE '%ONCO%'     OR specialty LIKE '%REUMATO%'
              OR specialty LIKE '%PNEUMO%'   OR specialty LIKE '%HEMATO%'
              OR specialty LIKE '%INFECTO%'  OR specialty LIKE '%ALERGO%'
              OR specialty LIKE '%OTORRINO%' OR specialty LIKE '%NUTRI%'
              OR specialty LIKE '%HEPATO%'   OR specialty LIKE '%GLAUCOMA%'
              THEN 'CLINICA_ESPECIALIZADA'
            ELSE 'OUTRAS_ESPECIALIDADES'
        END AS specialty_group
    FROM temporal
)"""

    # ─── CTE 4: Window functions para histórico do paciente (raw) ─────────────
    hist_raw_cte = """hist_raw AS (
    SELECT *,
        -- Etapa 7: histórico — anti-leakage via ROWS UNBOUNDED PRECEDING AND 1 PRECEDING
        (ROW_NUMBER() OVER (
            PARTITION BY patient_id ORDER BY appointment_at, appointment_id
        ) - 1)::INT AS previous_appointments_count,
        -- Equivalente exato de pandas (ts2-ts1).days = floor(segundos / 86400)
        -- Não usar DATE_TRUNC pois pandas usa diff de timestamp, não de data de calendário
        GREATEST(0, FLOOR(DATEDIFF('second',
            MIN(appointment_at) OVER (PARTITION BY patient_id),
            appointment_at
        ) / 86400.0))::INT AS patient_tenure_days,
        COALESCE(SUM(is_canceled_temp) OVER (
            PARTITION BY patient_id ORDER BY appointment_at, appointment_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0)::INT AS past_cancellations_count,
        COALESCE(SUM(COALESCE(no_show, 0)) OVER (
            PARTITION BY patient_id ORDER BY appointment_at, appointment_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ), 0)::INT AS past_no_shows,
        LAG(no_show, 1) OVER (
            PARTITION BY patient_id ORDER BY appointment_at, appointment_id
        ) AS _lag_ns_1,
        LAG(no_show, 2) OVER (
            PARTITION BY patient_id ORDER BY appointment_at, appointment_id
        ) AS _lag_ns_2,
        -- Rolling rates (equivalente ao shift(1).rolling(N, min_periods=1).mean())
        AVG(CAST(COALESCE(no_show, 0) AS FLOAT)) OVER (
            PARTITION BY patient_id ORDER BY appointment_at, appointment_id
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS _rate_3_raw,
        AVG(CAST(COALESCE(no_show, 0) AS FLOAT)) OVER (
            PARTITION BY patient_id ORDER BY appointment_at, appointment_id
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS _rate_5_raw,
        -- Data do último no-show (para days_since_last_no_show)
        MAX(CASE WHEN no_show = 1 THEN appointment_at ELSE NULL END) OVER (
            PARTITION BY patient_id ORDER BY appointment_at, appointment_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS _last_ns_date,
        -- Data da última consulta (para days_since_last_visit)
        LAG(appointment_at, 1) OVER (
            PARTITION BY patient_id ORDER BY appointment_at, appointment_id
        ) AS _last_visit_date
    FROM enriched
)"""

    # ─── CTE 5: Features de histórico derivadas dos windows ───────────────────
    hist_cte = """hist AS (
    SELECT *,
        (previous_appointments_count > 0)::TINYINT AS has_patient_history,
        CASE WHEN previous_appointments_count > 0
             THEN CAST(past_cancellations_count AS FLOAT) / previous_appointments_count
             ELSE 0.0
        END::FLOAT AS cancellation_rate,
        -- previous_no_show: -1 para primeira consulta, LAG(no_show) para demais
        CASE WHEN previous_appointments_count = 0 THEN -1
             ELSE COALESCE(_lag_ns_1, -1)
        END::TINYINT AS previous_no_show,
        -- consecutive_no_shows_2: -1 se < 2 consultas anteriores
        CASE
            WHEN previous_appointments_count < 2 THEN -1
            WHEN COALESCE(_lag_ns_1, 0) = 1 AND COALESCE(_lag_ns_2, 0) = 1 THEN 1
            ELSE 0
        END::TINYINT AS consecutive_no_shows_2,
        -- Taxa suavizada (Laplace smoothing): (past_ns + 1) / (count + 2)
        ((past_no_shows + 1.0) / (previous_appointments_count + 2.0))::FLOAT
            AS no_show_rate_patient_smoothed,
        CASE WHEN previous_appointments_count = 0 THEN -1.0
             ELSE (past_no_shows + 1.0) / (previous_appointments_count + 2.0)
        END::FLOAT AS no_show_rate_patient,
        -- Taxas recentes (rolling 3 e 5): -1 para primeira consulta
        CASE WHEN previous_appointments_count = 0 THEN -1.0
             ELSE COALESCE(_rate_3_raw, -1.0)
        END::FLOAT AS no_show_rate_recent_3,
        CASE WHEN previous_appointments_count = 0 THEN -1.0
             ELSE COALESCE(_rate_5_raw, -1.0)
        END::FLOAT AS no_show_rate_recent_5,
        CASE WHEN _last_ns_date IS NULL THEN -1.0
             ELSE CAST(DATEDIFF('day', _last_ns_date, appointment_at) AS FLOAT)
        END::FLOAT AS days_since_last_no_show,
        CASE WHEN _last_visit_date IS NULL THEN -1.0
             ELSE CAST(DATEDIFF('day', _last_visit_date, appointment_at) AS FLOAT)
        END::FLOAT AS days_since_last_visit
    FROM hist_raw
)"""

    # ─── CTE 6: Features comportamentais ─────────────────────────────────────
    behavioral_cte = """behavioral AS (
    SELECT *,
        -- Etapa 10: variação de lead time entre consultas
        COALESCE(
            CAST(waiting_days AS FLOAT)
            - CAST(LAG(waiting_days) OVER (
                PARTITION BY patient_id ORDER BY appointment_at, appointment_id
              ) AS FLOAT),
            0.0
        )::FLOAT AS waiting_days_delta,
        -- Mudança de especialidade em relação à consulta anterior
        CASE
            WHEN LAG(specialty) OVER (
                PARTITION BY patient_id ORDER BY appointment_at, appointment_id
            ) IS NOT NULL
             AND specialty != LAG(specialty) OVER (
                PARTITION BY patient_id ORDER BY appointment_at, appointment_id
            ) THEN 1
            ELSE 0
        END::TINYINT AS is_diff_specialty,
        -- Consultas agendadas no mesmo momento (batch booking)
        COUNT(*) OVER (
            PARTITION BY patient_id, scheduled_at
        )::SMALLINT AS appointments_in_same_schedule_day,
        -- Perfil demográfico combinado
        UPPER(TRIM(patient_sex)) || '_' || age_group AS gender_age_profile
    FROM hist
)"""

    # ─── CTE 7: Taxas contextuais (anti-leakage via expanding mean) ──────────
    contextual_cte = """contextual AS (
    SELECT *,
        -- Etapa 11: expanding mean com anti-leakage (shift=1 implícito via ROWS ... AND 1 PRECEDING)
        AVG(CAST(COALESCE(no_show, 0) AS FLOAT)) OVER (
            PARTITION BY unit_name ORDER BY appointment_at, appointment_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )::FLOAT AS unit_no_show_rate,
        AVG(CAST(COALESCE(no_show, 0) AS FLOAT)) OVER (
            PARTITION BY specialty ORDER BY appointment_at, appointment_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )::FLOAT AS specialty_no_show_rate,
        AVG(CAST(COALESCE(no_show, 0) AS FLOAT)) OVER (
            PARTITION BY specialty_group ORDER BY appointment_at, appointment_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )::FLOAT AS specialty_group_no_show_rate,
        AVG(CAST(COALESCE(no_show, 0) AS FLOAT)) OVER (
            PARTITION BY patient_neighborhood ORDER BY appointment_at, appointment_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )::FLOAT AS neighborhood_risk_score,
        AVG(CAST(COALESCE(no_show, 0) AS FLOAT)) OVER (
            PARTITION BY insurance_type ORDER BY appointment_at, appointment_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )::FLOAT AS insurance_no_show_rate
    FROM behavioral
)"""

    # ─── CTE 8: Flag de alto risco + features de interação ───────────────────
    interaction_cte = """interaction AS (
    SELECT *,
        -- Etapa 11: flag de especialidade de alto risco (threshold 0.35)
        CASE WHEN specialty_group_no_show_rate IS NULL THEN 0
             WHEN specialty_group_no_show_rate >= 0.35  THEN 1
             ELSE 0
        END::TINYINT AS specialty_high_no_show_flag,
        -- Etapa 12: features de interação
        (CAST(patient_age AS FLOAT) * CAST(waiting_days AS FLOAT))::FLOAT
            AS age_x_waiting_days,
        (GREATEST(0.0, COALESCE(no_show_rate_patient, 0.0))
            * CAST(waiting_days AS FLOAT))::FLOAT
            AS no_show_rate_x_waiting_days,
        (CAST(patient_age AS FLOAT)
            * GREATEST(0.0, COALESCE(specialty_group_no_show_rate, 0.0)))::FLOAT
            AS age_x_specialty_risk
    FROM contextual
)"""

    # ─── CTE 9: Feriados ──────────────────────────────────────────────────────
    holidays_cte = """holidays_base AS (
    -- Etapa 13: feriados via LEFT JOIN com tabela registrada br_holidays
    SELECT t.*,
        CASE WHEN h_today.holiday_date IS NOT NULL THEN 1 ELSE 0 END::TINYINT AS is_holiday,
        CASE WHEN h_pre.holiday_date   IS NOT NULL THEN 1 ELSE 0 END::TINYINT AS is_pre_holiday,
        CASE WHEN h_post.holiday_date  IS NOT NULL THEN 1 ELSE 0 END::TINYINT AS is_post_holiday
    FROM interaction t
    LEFT JOIN br_holidays h_today ON CAST(t.appointment_at AS DATE)     = h_today.holiday_date
    LEFT JOIN br_holidays h_pre   ON CAST(t.appointment_at AS DATE) + 1 = h_pre.holiday_date
    LEFT JOIN br_holidays h_post  ON CAST(t.appointment_at AS DATE) - 1 = h_post.holiday_date
),
holidays_added AS (
    SELECT *,
        -- Ponte: sexta antes de feriado (weekday=4) ou segunda após feriado (weekday=0)
        CASE WHEN (appointment_weekday = 4 AND is_pre_holiday = 1)
              OR  (appointment_weekday = 0 AND is_post_holiday = 1)
             THEN 1 ELSE 0
        END::TINYINT AS is_bridge_day
    FROM holidays_base
),
holidays_final AS (
    SELECT *,
        GREATEST(is_holiday, is_pre_holiday, is_post_holiday, is_bridge_day)::TINYINT
            AS is_holiday_window
    FROM holidays_added
)"""

    # ─── CTE 10: Features geográficas ─────────────────────────────────────────
    geo_cte = """geo_raw AS (
    SELECT *,
        -- Etapa 14: mesmo prefixo CEP (5 dígitos)
        CASE
            WHEN patient_cep IS NULL OR unit_cep IS NULL THEN -1
            WHEN LENGTH(REGEXP_REPLACE(patient_cep, '[^0-9]', '', 'g')) < 5
              OR LENGTH(REGEXP_REPLACE(unit_cep,    '[^0-9]', '', 'g')) < 5 THEN -1
            WHEN LEFT(REGEXP_REPLACE(patient_cep, '[^0-9]', '', 'g'), 5)
               = LEFT(REGEXP_REPLACE(unit_cep,    '[^0-9]', '', 'g'), 5)
            THEN 1 ELSE 0
        END::TINYINT AS same_cep_prefix5,
        -- Mesma cidade: compara patient_city com segmento após último '-' do unit_address
        CASE
            WHEN patient_city IS NULL OR unit_address IS NULL THEN 0
            WHEN UPPER(TRIM(patient_city))
               = UPPER(TRIM(list_last(string_split(unit_address, '-'))))
            THEN 1 ELSE 0
        END::TINYINT AS same_city
    FROM holidays_final
),
geo_final AS (
    SELECT *,
        same_city::TINYINT AS is_local_resident
    FROM geo_raw
)"""

    # ─── CTE 11: Normalização de texto ────────────────────────────────────────
    normalized_cte = f"""normalized AS (
    -- Etapa 15: uppercase + trim de colunas de texto
    -- Seleciona explicitamente para descartar colunas temporárias (_lag_*, _rate_*, etc.)
    SELECT
        appointment_id,
        patient_id,
        UPPER(TRIM(appointment_status)) AS appointment_status,
        no_show,
        scheduled_at,
        appointment_at,
        waiting_days,
        is_same_day,
        appointment_weekday,
        is_weekend,
        appointment_day_of_month,
        appointment_week_of_month,
        is_month_start,
        is_month_end,
        hour_appointment,
        time_of_day,
        month_sin,
        month_cos,
        weekday_sin,
        weekday_cos,
        hour_sin,
        hour_cos,
        is_holiday,
        is_pre_holiday,
        is_post_holiday,
        is_bridge_day,
        is_holiday_window,
        UPPER(TRIM(patient_sex))          AS patient_sex,
        patient_age,
        age_group,
        UPPER(TRIM(insurance_type))       AS insurance_type,
        UPPER(TRIM(patient_city))         AS patient_city,
        UPPER(TRIM(patient_neighborhood)) AS patient_neighborhood,
        UPPER(TRIM(unit_name))            AS unit_name,
        specialty,
        specialty_group,
        UPPER(TRIM(unit_address))         AS unit_address,
        UPPER(TRIM(unit_cep))             AS unit_cep,
        has_patient_history,
        previous_appointments_count,
        patient_tenure_days,
        past_no_shows,
        previous_no_show,
        consecutive_no_shows_2,
        past_cancellations_count,
        cancellation_rate,
        no_show_rate_patient,
        no_show_rate_patient_smoothed,
        no_show_rate_recent_3,
        no_show_rate_recent_5,
        days_since_last_visit,
        days_since_last_no_show,
        appointments_in_same_schedule_day,
        is_diff_specialty,
        waiting_days_delta,
        age_x_waiting_days,
        no_show_rate_x_waiting_days,
        age_x_specialty_risk,
        UPPER(TRIM(gender_age_profile))   AS gender_age_profile,
        unit_no_show_rate,
        specialty_no_show_rate,
        specialty_group_no_show_rate,
        insurance_no_show_rate,
        neighborhood_risk_score,
        specialty_high_no_show_flag,
        same_city,
        is_local_resident,
        same_cep_prefix5,
        is_new_row
    FROM geo_final
)"""

    # ─── CTE 12: Filtro final + ordenação de colunas ─────────────────────────
    final_cte = f"""final AS (
    -- Etapa 16: manter apenas linhas novas + reordenar colunas conforme config
    SELECT
        {final_cols}
    FROM normalized
    WHERE is_new_row = TRUE
)"""

    # ─── Montar SQL completo ──────────────────────────────────────────────────
    ctes = [
        input_cte,
        base_cte,
        temporal_cte,
        enriched_cte,
        hist_raw_cte,
        hist_cte,
        behavioral_cte,
        contextual_cte,
        interaction_cte,
        holidays_cte,
        geo_cte,
        normalized_cte,
        final_cte,
    ]

    # DuckDB exige que COPY envolva a query inteira como subquery:
    # COPY (WITH ... SELECT * FROM final) TO '...' (FORMAT PARQUET)
    cte_block = "WITH\n" + ",\n".join(ctes)
    full_sql = f"COPY (\n{cte_block}\nSELECT * FROM final\n) TO '{p_out}' (FORMAT PARQUET);"
    return full_sql
