# run_pipeline.py

import logging
from pathlib import Path
import pandas as pd

# Import your library modules
from noshow_lib.config import load_config
from noshow_lib.data_processing import load_and_process_data
from noshow_lib.feature_engineering import build_features

# Configure logging to show messages in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Main execution flow:
    1. Load Configuration.
    2. ETL (Load & Clean Data).
    3. Feature Engineering.
    4. Save Final Dataset.
    """
    
    # =================================================================
    # 1. SETUP PATHS & CONFIG
    # =================================================================
    project_root = Path(__file__).parent
    config_path = project_root / "config.yaml"
    
    logger.info(f"Loading configuration from: {config_path}")
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.critical(f"Failed to load config: {e}")
        return

    # Extract paths from YAML (making them absolute based on project root)
    # YAML says: "data/01 - raw/noshowappointments.csv"
    raw_rel_path = config["data"]["raw_data_path"]
    processed_rel_path = config["data"]["processed_data_path"]
    
    raw_path = project_root / raw_rel_path
    processed_path = project_root / processed_rel_path

    # Define where to save the FINAL file (with features)
    # You can add this to YAML later, but for now let's define here
    features_path = project_root / "data" / "03 - features" / "dataset_final.csv"

    # =================================================================
    # 2. DATA PROCESSING (Clean & Filter)
    # =================================================================
    logger.info("Starting Data Processing Stage...")
    
    try:
        # This function loads CSV, cleans it, and saves the intermediate file
        df_clean = load_and_process_data(
            input_data=raw_path,
            processed_path=processed_path,
            config=config.get("preprocessing")
        )
    except Exception as e:
        logger.critical(f"Data Processing failed: {e}")
        return

    # =================================================================
    # 3. FEATURE ENGINEERING (Enrich Data)
    # =================================================================
    logger.info("Starting Feature Engineering Stage...")
    
    try:
        # We pass the 'data' block because it contains 'target_column'
        df_final = build_features(
            df=df_clean, 
            data_cfg=config.get("data")
        )
    except Exception as e:
        logger.critical(f"Feature Engineering failed: {e}")
        return

    # =================================================================
    # 4. SAVE FINAL DATASET
    # =================================================================
    logger.info("Saving Final Dataset...")
    
    try:
        # Ensure directory exists
        features_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_final.to_csv(features_path, index=False)
        logger.info(f"Pipeline finished successfully!")
        logger.info(f"Final file saved at: {features_path}")
        logger.info(f"Final Shape: {df_final.shape}")
        
    except Exception as e:
        logger.error(f"Failed to save final file: {e}")

if __name__ == "__main__":
    main()