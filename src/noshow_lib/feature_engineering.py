# src/noshow_lib/feature_engineering.py

import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def build_features(df: pd.DataFrame, data_cfg: Dict) -> pd.DataFrame:
    """
    Applies feature engineering logic to the DataFrame.

    Key transformations:
    1. Date processing (waiting days, hour, etc).
    2. Patient history (previous appointments).
    3. Age binning.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        data_cfg (dict): Configuration dict containing 'target_column'.

    Returns:
        pd.DataFrame: DataFrame with new features.
    
    Raises:
        ValueError: If target column is missing in config.
        KeyError: If expected columns are missing in the DataFrame.
    """
    logger.info("Starting feature engineering...")

    # -------------------------------------------------------------------
    # 1. Get target column from YAML config
    # -------------------------------------------------------------------
    try:
        target_column = data_cfg.get("target_column")
        if target_column is None:
            raise ValueError("YAML config missing 'data.target_column'.")
        
        # Renaming target column for internal standardization
        # Using copy() to avoid SettingWithCopyWarning
        df_temp = df.copy().rename(columns={target_column: "No_show"})
        
    except Exception as e:
        logger.error(f"Error initializing feature engineering: {e}")
        raise

    # -------------------------------------------------------------------
    # 2. Cleaning and Type Conversions
    # -------------------------------------------------------------------
    try:
        # Check if required columns exist before processing
        required_cols = ["ScheduledDay", "AppointmentDay", "Age", "No_show"]
        missing = [col for col in required_cols if col not in df_temp.columns]
        if missing:
            raise KeyError(f"Missing columns for feature engineering: {missing}")

        df_temp = (
            df_temp
            .assign(
                No_show=lambda x: x["No_show"].map({"No": 0, "Yes": 1}).astype(int),
                ScheduledDay=lambda x: pd.to_datetime(x["ScheduledDay"], errors="coerce", utc=True),
                AppointmentDay=lambda x: pd.to_datetime(x["AppointmentDay"], errors="coerce", utc=True),
            )
            .query("Age >= 0")
            .dropna(subset=["ScheduledDay", "AppointmentDay"])
        )
    except Exception as e:
        logger.error(f"Error during data cleaning/conversion: {e}")
        raise

    # -------------------------------------------------------------------
    # 3. Temporal Features (Date/Time)
    # -------------------------------------------------------------------
    try:
        df_temp = df_temp.assign(
            waiting_days=lambda x: (x["AppointmentDay"] - x["ScheduledDay"]).dt.days,
            appointment_weekday=lambda x: x["AppointmentDay"].dt.weekday,
            appointment_month=lambda x: x["AppointmentDay"].dt.month,
            hour_scheduled=lambda x: x["ScheduledDay"].dt.hour,
            hour_appointment=lambda x: x["AppointmentDay"].dt.hour,
            is_weekend=lambda x: x["AppointmentDay"].dt.weekday.isin([5, 6]).astype(int),
        )
    except Exception as e:
        logger.error(f"Error creating temporal features: {e}")
        raise

    # -------------------------------------------------------------------
    # 4. Patient History (Previous Appointments)
    # -------------------------------------------------------------------
    try:
        if "PatientId" in df_temp.columns and "AppointmentID" in df_temp.columns:
            # Note: Ensure data is sorted by date if you want true historical count
            df_temp["previous_appointments"] = (
                df_temp.groupby("PatientId")["AppointmentID"].transform("count") - 1
            )
        else:
            logger.warning("PatientId or AppointmentID missing. Skipping 'previous_appointments'.")
            
    except Exception as e:
        logger.error(f"Error calculating previous appointments: {e}")
        raise

    # -------------------------------------------------------------------
    # 5. Patient No-show Rate
    # -------------------------------------------------------------------
    try:
        # Note: Be careful with Data Leakage here (using mean of target)
        if "PatientId" in df_temp.columns:
            df_temp["no_show_rate_patient"] = (
                df_temp.groupby("PatientId")["No_show"].transform("mean")
            )
    except Exception as e:
        logger.error(f"Error calculating no-show rate: {e}")
        raise

    # -------------------------------------------------------------------
    # 6. Age Group Binning
    # -------------------------------------------------------------------
    try:
        if "Age" in df_temp.columns:
            # Using max() + 1 to ensure the highest value is included
            max_age = df_temp["Age"].max()
            age_bins = [0, 1, 13, 18, 60, max_age + 1]
            age_labels = ["baby", "child", "teen", "adult", "senior"]

            df_temp["age_group"] = pd.cut(
                df_temp["Age"],
                bins=age_bins,
                labels=age_labels,
                right=False
            ).astype(str)
            
    except Exception as e:
        logger.error(f"Error creating age groups: {e}")
        raise

    # -------------------------------------------------------------------
    # 7. Rename target back to original
    # -------------------------------------------------------------------
    try:
        df_temp = df_temp.rename(columns={"No_show": target_column})
    except Exception as e:
        logger.error(f"Error restoring target column name: {e}")
        raise

    logger.info(f"Feature engineering completed. Final shape: {df_temp.shape}")
    return df_temp