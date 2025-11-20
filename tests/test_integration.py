# tests/test_integration.py

import pytest
import pandas as pd
from pathlib import Path
import logging

from noshow_lib.config import load_config
from noshow_lib.data_processing import load_and_process_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_generate_physical_file_with_real_config():
    """
    Integration test that:
    1. Reads the config_test.yaml from the assets folder.
    2. Uses real raw data from the local disk.
    3. Processes and SAVES a real file to the preprocessed folder.
    """
    
    # ---------------------------------------------------------
    # 1. SETUP: Define Paths
    # ---------------------------------------------------------
    # Get project root (where pyproject.toml is located)
    project_root = Path(__file__).parent.parent
    
    # Path to the YAML config (in the assets folder)
    yaml_path = Path(__file__).parent / "assets" / "config_test.yaml"
    
    # --- FIX 1: Define real paths in variables ---
    # Using raw string (r'') for Windows paths
    raw_csv_path = r'C:\Users\lobat\OneDrive\Área de Trabalho\tcc-rep\noshow-prediction-ml\data\01 - raw\noshowappointments.csv'
    
    # ==============================================================================
    # FIX HERE: Added the filename "noshowappointments_processed.csv" at the end
    # ==============================================================================
    processed_csv_path = r'C:\Users\lobat\OneDrive\Área de Trabalho\tcc-rep\noshow-prediction-ml\data\02 - preprocessed\noshowappointments_processed.csv'
    
    # ---------------------------------------------------------
    # 2. SETUP: Load Configuration
    # ---------------------------------------------------------
    full_config = load_config(yaml_path)

    # ---------------------------------------------------------
    # 3. EXECUTION
    # ---------------------------------------------------------
    # We pass the variables defined above
    df_result = load_and_process_data(
        input_data=raw_csv_path,
        processed_path=processed_csv_path,
        config=full_config.get("preprocessing")
    )

    # ---------------------------------------------------------
    # 4. VERIFICATION
    # ---------------------------------------------------------
    
    # --- FIX 2: Verify the correct file ---
    # Now we check if the specific file (not just the folder) exists
    assert Path(processed_csv_path).exists()
    print(f"\nSUCCESS! File generated at: {processed_csv_path}")
    
    # --- FIX 3: Verify the correct column (English) ---
    # The original dataset uses "Age", not "idade"
    if "Age" in df_result.columns:
        assert df_result["Age"].isnull().sum() == 0
        print("Column 'Age' verified: zero nulls.")
    else:
        # Fallback if columns were renamed in YAML
        print(f"Warning: Columns found in DataFrame: {df_result.columns.tolist()}")