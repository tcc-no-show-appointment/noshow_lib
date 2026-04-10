import pandas as pd
import numpy as np
from datetime import date
from typing import Optional
from .logger import setup_logger

logger = setup_logger("noshow_lib.stats_enrichment")

# Columns that indicate a patient has valid precomputed history
_PATIENT_HISTORY_SIGNAL_COLS = [
    "previous_appointments_count",
    "past_no_shows",
    "no_show_rate_patient",
]

# All patient history columns that get merged
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
    "past_cancellations_count",
    "has_patient_history",
]

# Mapping: dimension column → rate column name
_CONTEXTUAL_DIMENSIONS = {
    "unit_name": "unit_no_show_rate",
    "specialty": "specialty_no_show_rate",
    "specialty_group": "specialty_group_no_show_rate",
    "insurance_type": "insurance_no_show_rate",
    "patient_neighborhood": "neighborhood_risk_score",
}


def enrich_with_precomputed_stats(
    df: pd.DataFrame,
    patient_stats: Optional[pd.DataFrame],
    contextual_stats: Optional[pd.DataFrame],
    config: dict,
    reference_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Merge precomputed stats into the input DataFrame before build_features().

    Only enriches rows that have a patient_id. Rows without patient_id are
    left untouched (they will get default -1 values during build_features).

    Args:
        df: Input DataFrame (raw appointment data, before or after column rename).
        patient_stats: Output of precompute_patient_stats(). Can be None.
        contextual_stats: Output of precompute_contextual_stats(). Can be None.
        config: Config dict (YAML) — used to resolve column_map.
        reference_date: Date of prediction. Used to validate cutoff and
                        recompute days_since_last_visit. Defaults to today.

    Returns:
        DataFrame enriched with precomputed columns.
    """
    if patient_stats is None and contextual_stats is None:
        logger.info("No precomputed stats provided. Skipping enrichment.")
        return df

    df = df.copy()
    reference_date = reference_date or date.today()

    # Resolve patient_id column name (could be raw "idUnicoPaciente" or internal "patient_id")
    column_map = config.get("column_map", {})
    reverse_map = {v: k for k, v in column_map.items()}

    patient_id_col = _resolve_column(df, "patient_id", column_map, reverse_map)

    if patient_id_col is None:
        logger.info("No patient ID column found in input. Skipping enrichment.")
        return df

    logger.info(f"Enriching {len(df)} rows using patient_id column: '{patient_id_col}'")

    # ── Patient stats merge ────────────────────────────────────────────
    if patient_stats is not None and not patient_stats.empty:
        df = _merge_patient_stats(df, patient_stats, patient_id_col, reference_date)

    # ── Contextual stats merge ─────────────────────────────────────────
    if contextual_stats is not None and not contextual_stats.empty:
        df = _merge_contextual_stats(df, contextual_stats, column_map, reverse_map)

    return df


def _resolve_column(
    df: pd.DataFrame,
    internal_name: str,
    column_map: dict,
    reverse_map: dict,
) -> Optional[str]:
    """Find which column name (raw or internal) exists in df for a given internal name."""
    if internal_name in df.columns:
        return internal_name
    raw_name = reverse_map.get(internal_name)
    if raw_name and raw_name in df.columns:
        return raw_name
    return None


def _merge_patient_stats(
    df: pd.DataFrame,
    patient_stats: pd.DataFrame,
    patient_id_col: str,
    reference_date: date,
) -> pd.DataFrame:
    """Merge patient-level stats into df by patient_id."""
    # Validate cutoff
    if "stats_cutoff_date" in patient_stats.columns:
        cutoff_str = patient_stats["stats_cutoff_date"].iloc[0]
        cutoff = pd.to_datetime(cutoff_str).date()
        if cutoff >= reference_date:
            logger.warning(
                f"LEAKAGE RISK: stats_cutoff_date={cutoff} >= reference_date={reference_date}. "
                f"Stats may contain future data. Proceeding with caution."
            )

    # Prepare stats for merge (join key = patient_id)
    # Support both indexed (patient_id as index) and flat DataFrames
    if patient_stats.index.name == "patient_id":
        stats_to_merge = patient_stats.reset_index().drop(
            columns=["stats_cutoff_date"], errors="ignore"
        ).copy()
    else:
        stats_to_merge = patient_stats.drop(
            columns=["stats_cutoff_date"], errors="ignore"
        ).copy()

    # Rename patient_id to match the input column name
    if patient_id_col != "patient_id" and "patient_id" in stats_to_merge.columns:
        stats_to_merge = stats_to_merge.rename(columns={"patient_id": patient_id_col})

    available_cols = [c for c in _PATIENT_HISTORY_COLS if c in stats_to_merge.columns]
    merge_cols = [patient_id_col] + available_cols
    if "last_appointment_at" in stats_to_merge.columns:
        merge_cols.append("last_appointment_at")

    stats_to_merge = stats_to_merge[[c for c in merge_cols if c in stats_to_merge.columns]]

    before_cols = set(df.columns)
    df = df.merge(stats_to_merge, on=patient_id_col, how="left")

    # Recalculate days_since_last_visit using the actual reference_date
    if "last_appointment_at" in df.columns:
        last_appt = pd.to_datetime(df["last_appointment_at"], errors="coerce")
        df["days_since_last_visit"] = (
            (pd.Timestamp(reference_date) - last_appt).dt.days
            .fillna(-1)
            .astype("float32")
        )
        df.drop(columns=["last_appointment_at"], inplace=True)

    # Fill NaN for patients not found in stats (new patients)
    for col in available_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    # Set has_patient_history based on merged data
    if "previous_appointments_count" in df.columns:
        df["has_patient_history"] = (df["previous_appointments_count"] > 0).astype("int8")

    new_cols = set(df.columns) - before_cols
    matched = df[patient_id_col].isin(stats_to_merge[patient_id_col]).sum()
    logger.info(
        f"Patient stats merged: {matched}/{len(df)} rows matched. "
        f"New columns: {sorted(new_cols)}"
    )

    return df


def _merge_contextual_stats(
    df: pd.DataFrame,
    contextual_stats: pd.DataFrame,
    column_map: dict,
    reverse_map: dict,
) -> pd.DataFrame:
    """Merge contextual rates into df by each dimension."""
    stats = contextual_stats.drop(columns=["stats_cutoff_date"], errors="ignore")

    merged_count = 0
    for dim_internal, rate_col in _CONTEXTUAL_DIMENSIONS.items():
        dim_rows = stats[stats["dimension"] == dim_internal]
        if dim_rows.empty:
            continue

        # Find which column name exists in df
        dim_col_in_df = _resolve_column(df, dim_internal, column_map, reverse_map)
        if dim_col_in_df is None:
            continue

        # Build lookup: dimension_value → rate
        lookup = (
            dim_rows[["dimension_value", "rate"]]
            .rename(columns={"dimension_value": dim_col_in_df, "rate": rate_col})
        )
        lookup[dim_col_in_df] = lookup[dim_col_in_df].astype(str).str.strip().str.upper()

        # Normalize df column for matching
        df_dim_normalized = df[dim_col_in_df].astype(str).str.strip().str.upper()
        temp_col = f"__{dim_col_in_df}_norm"
        df[temp_col] = df_dim_normalized
        lookup = lookup.rename(columns={dim_col_in_df: temp_col})

        df = df.merge(lookup, on=temp_col, how="left")
        df.drop(columns=[temp_col], inplace=True)

        df[rate_col] = df[rate_col].fillna(-1.0).astype("float32")
        merged_count += 1

    # specialty_high_no_show_flag derived from specialty_group_no_show_rate
    if "specialty_group_no_show_rate" in df.columns:
        df["specialty_high_no_show_flag"] = np.where(
            df["specialty_group_no_show_rate"] < 0, 0,
            (df["specialty_group_no_show_rate"] >= 0.35).astype("int8"),
        ).astype("int8")

    logger.info(f"Contextual stats merged for {merged_count} dimensions.")
    return df
