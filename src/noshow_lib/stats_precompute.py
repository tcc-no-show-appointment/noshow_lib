import pandas as pd
import duckdb
from datetime import date
from pathlib import Path
from typing import Union, Optional
from .logger import setup_logger

logger = setup_logger("noshow_lib.stats_precompute")

# Patient-level columns extracted from build_features / build_features_from_parquet output
_PATIENT_HISTORY_COLS = [
    "previous_appointments_count",
    "past_no_shows",
    "no_show_rate_patient",
    "no_show_rate_patient_smoothed",
    "no_show_rate_recent_3",
    "no_show_rate_recent_5",
    "previous_no_show",
    "consecutive_no_shows_2",
    "cancellation_rate",
    "patient_tenure_days",
    "days_since_last_no_show",
    "days_since_last_visit",
    "past_cancellations_count",
    "has_patient_history",
]

# Contextual dimensions → output column name
_CONTEXTUAL_DIMENSIONS = {
    "unit_name": "unit_no_show_rate",
    "specialty": "specialty_no_show_rate",
    "specialty_group": "specialty_group_no_show_rate",
    "insurance_type": "insurance_no_show_rate",
    "patient_neighborhood": "neighborhood_risk_score",
}


def _esc(s: str) -> str:
    """Escape a path for DuckDB SQL string literals."""
    return s.replace("'", "''").replace("\\", "/")


def _resolve_source(con: duckdb.DuckDBPyConnection, source) -> str:
    """
    Register source (DataFrame or parquet path) and return SQL table reference.

    - pd.DataFrame → registered as '_src_data' view (zero-copy)
    - str / Path   → read_parquet('{path}')
    """
    if isinstance(source, pd.DataFrame):
        con.register("_src_data", source)
        return "_src_data"
    else:
        path = _esc(str(source))
        return f"read_parquet('{path}')"


def _get_available_columns(con: duckdb.DuckDBPyConnection, table_ref: str) -> set:
    """Get column names available in the source."""
    rows = con.execute(f"DESCRIBE SELECT * FROM {table_ref} LIMIT 0").fetchall()
    return {row[0] for row in rows}


def precompute_patient_stats(
    source: Union[pd.DataFrame, str, Path],
    cutoff_date: date,
) -> pd.DataFrame:
    """
    Extract per-patient aggregated stats via DuckDB.

    Uses the last row per patient (chronologically), which already contains
    anti-leakage historical features (shift(1) applied during feature engineering).

    Args:
        source: Features DataFrame, or path to features Parquet file
                (output of build_features / build_features_from_parquet).
        cutoff_date: Only rows with appointment_at < cutoff_date are considered.

    Returns:
        DataFrame with one row per patient_id containing historical stats.
    """
    con = duckdb.connect()
    table_ref = _resolve_source(con, source)

    # Detect available columns
    available = _get_available_columns(con, table_ref)
    if "patient_id" not in available or "appointment_at" not in available:
        raise ValueError("Source must contain 'patient_id' and 'appointment_at' columns.")

    # Build SELECT list from available history columns
    history_selects = [c for c in _PATIENT_HISTORY_COLS if c in available]
    if not history_selects:
        raise ValueError("No patient history columns found in source.")

    cols_sql = ",\n            ".join(history_selects)
    cutoff_str = cutoff_date.isoformat()

    sql = f"""
        WITH cutoff_data AS (
            SELECT *
            FROM {table_ref}
            WHERE CAST(appointment_at AS TIMESTAMP) < CAST('{cutoff_str}' AS TIMESTAMP)
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY patient_id
                    ORDER BY appointment_at DESC, appointment_id DESC
                ) AS _rn
            FROM cutoff_data
        )
        SELECT
            CAST(patient_id AS VARCHAR) AS patient_id,
            appointment_at AS last_appointment_at,
            {cols_sql},
            '{cutoff_str}' AS stats_cutoff_date
        FROM ranked
        WHERE _rn = 1
    """

    # Fallback: if appointment_id doesn't exist, adjust ORDER BY
    if "appointment_id" not in available:
        sql = sql.replace(
            "ORDER BY appointment_at DESC, appointment_id DESC",
            "ORDER BY appointment_at DESC"
        )

    patient_stats = con.execute(sql).df()
    con.close()

    logger.info(
        f"Patient stats generated via DuckDB: {len(patient_stats)} patients, "
        f"{len(history_selects)} feature columns (cutoff={cutoff_date})"
    )
    return patient_stats


def precompute_contextual_stats(
    source: Union[pd.DataFrame, str, Path],
    cutoff_date: date,
) -> pd.DataFrame:
    """
    Compute contextual no-show rates per dimension via DuckDB with Laplace smoothing.

    Args:
        source: Features DataFrame, or path to features Parquet file
                (output of build_features / build_features_from_parquet).
        cutoff_date: Only rows with appointment_at < cutoff_date are considered.

    Returns:
        Normalized DataFrame with columns:
        [dimension, rate_name, dimension_value, rate, total_count, stats_cutoff_date]
    """
    con = duckdb.connect()
    table_ref = _resolve_source(con, source)

    available = _get_available_columns(con, table_ref)
    if "no_show" not in available:
        raise ValueError("Source must contain 'no_show' column.")

    cutoff_str = cutoff_date.isoformat()
    union_parts = []

    for dim_col, rate_col in _CONTEXTUAL_DIMENSIONS.items():
        if dim_col not in available:
            logger.warning(f"Dimension column '{dim_col}' not found in source. Skipping.")
            continue

        union_parts.append(f"""
            SELECT
                '{dim_col}' AS dimension,
                '{rate_col}' AS rate_name,
                CAST({dim_col} AS VARCHAR) AS dimension_value,
                CAST((SUM(COALESCE(no_show, 0)) + 1.0) / (COUNT(*) + 2.0) AS FLOAT) AS rate,
                CAST(COUNT(*) AS INTEGER) AS total_count,
                '{cutoff_str}' AS stats_cutoff_date
            FROM {table_ref}
            WHERE CAST(appointment_at AS TIMESTAMP) < CAST('{cutoff_str}' AS TIMESTAMP)
              AND no_show IS NOT NULL
            GROUP BY {dim_col}
        """)

    if not union_parts:
        logger.warning("No contextual stats could be computed.")
        con.close()
        return pd.DataFrame()

    full_sql = "\nUNION ALL\n".join(union_parts)
    result = con.execute(full_sql).df()
    con.close()

    logger.info(
        f"Contextual stats generated via DuckDB: {len(result)} rows across "
        f"{result['dimension'].nunique()} dimensions (cutoff={cutoff_date})"
    )
    return result
