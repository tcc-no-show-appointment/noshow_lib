# tests/test_feature_engineering.py

import pytest
import pandas as pd
import numpy as np
from noshow_lib.feature_engineering import build_features

# =================================================================
# 1. FIXTURES (Prepared Data)
# =================================================================

@pytest.fixture
def sample_config():
    return {"target_column": "No-show"}

@pytest.fixture
def df_raw_features():
    """
    Creates a complex DataFrame to stress test the function.
    Includes:
    - Unsorted dates (to test internal sorting).
    - Data errors (Appointment before Schedule).
    - Weekends.
    """
    data = {
        "PatientId": [100, 100, 200, 300], 
        "AppointmentID": [2, 1, 3, 4], # IDs mixed on purpose
        "Age": [25, 25, 80, 5], 
        "No-show": ["Yes", "No", "No", "No"],
        
        "ScheduledDay": [
            "2023-05-10 10:00:00", # Patient 100, Visit 2 (Appears BEFORE Visit 1 in raw DF)
            "2023-05-01 08:00:00", # Patient 100, Visit 1 (Appears AFTER Visit 2 in raw DF)
            "2023-05-05 09:00:00", 
            "2023-05-20 10:00:00"  
        ],
        "AppointmentDay": [
            "2023-05-10 10:00:00", # Same day (Waiting = 0)
            "2023-05-05 08:00:00", # Normal difference
            "2023-05-06 14:00:00", # Saturday (Weekend)
            "2023-05-19 10:00:00"  # ERROR: Appointment before schedule (Negative waiting?)
        ]
    }
    return pd.DataFrame(data)

# =================================================================
# 2. BUSINESS LOGIC TESTS
# =================================================================

def test_historical_counts_sorting_logic(df_raw_features, sample_config):
    """
    CRITICAL TEST: Verifies if the function sorts data before calculating history.
    In the input, the visit on May 10th appears before the visit on May 5th.
    The function must fix this internally to count history correctly.
    """
    df_processed = build_features(df_raw_features, sample_config)
    
    # Filter only patient 100
    patient_100 = df_processed[df_processed["PatientId"] == 100]
    
    # The first CHRONOLOGICAL visit (May 5th) must have count 0
    # Even though it was on the second row of the original dataframe
    visit_day_5 = patient_100[patient_100["AppointmentDay"].dt.day == 5]
    assert visit_day_5.iloc[0]["previous_appointments_count"] == 0

    # The second CHRONOLOGICAL visit (May 10th) must have count 1
    visit_day_10 = patient_100[patient_100["AppointmentDay"].dt.day == 10]
    assert visit_day_10.iloc[0]["previous_appointments_count"] == 1

def test_waiting_days_logic_and_clipping(df_raw_features, sample_config):
    """
    Tests day calculation and prevents negative numbers (clipping).
    """
    df_processed = build_features(df_raw_features, sample_config)
    
    # Case 1: Scheduled 10th, Appointment 10th -> 0 days
    # Note: Filtering by ID to ensure we get the correct row
    row_same_day = df_processed[df_processed["AppointmentID"] == 2].iloc[0]
    assert row_same_day["waiting_days"] == 0
    
    # Case 2: Data Error (Scheduled 20th, Appointment 19th) -> Should be 0, not -1
    row_error = df_processed[df_processed["AppointmentID"] == 4].iloc[0]
    assert row_error["waiting_days"] == 0  # Thanks to .clip(lower=0)

def test_temporal_features_extraction(df_raw_features, sample_config):
    """Tests if weekend and hours are extracted correctly."""
    df_processed = build_features(df_raw_features, sample_config)
    
    # Patient 200 went on 06/05/2023 (Saturday)
    patient_saturday = df_processed[df_processed["PatientId"] == 200].iloc[0]
    
    assert patient_saturday["is_weekend"] == 1
    assert patient_saturday["appointment_weekday"] == 5 # 5 = Saturday
    assert patient_saturday["hour_appointment"] == 14 # It was at 14:00

def test_age_group_bins_completeness(df_raw_features, sample_config):
    """Tests age categories."""
    df_processed = build_features(df_raw_features, sample_config)
    
    # 80 years -> senior
    row_senior = df_processed[df_processed["Age"] == 80].iloc[0]
    assert row_senior["age_group"] == "senior"
    
    # 5 years -> child
    row_child = df_processed[df_processed["Age"] == 5].iloc[0]
    assert row_child["age_group"] == "child"

def test_error_missing_config():
    """Tests if the function raises an error if config is empty/bad."""
    df_empty = pd.DataFrame({"A": [1]})
    bad_config = {} # Missing target_column
    
    with pytest.raises(ValueError) as excinfo:
        build_features(df_empty, bad_config)
    
    assert "missing 'data.target_column'" in str(excinfo.value)

def test_target_encoding_is_numeric(df_raw_features, sample_config):
    """Ensures target becomes numeric (0/1) and not string."""
    df_processed = build_features(df_raw_features, sample_config)
    
    assert pd.api.types.is_numeric_dtype(df_processed["No-show"])
    assert set(df_processed["No-show"].unique()).issubset({0, 1})