# run_pipeline.py

import logging
import sys
from pathlib import Path
import pandas as pd

# Import library modules
from noshow_lib.config import load_config
from noshow_lib.data_processing import load_and_process_data
from noshow_lib.feature_engineering import build_features
from noshow_lib.model_training import train_model  # New import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    End-to-End Pipeline Execution:
    1. Load Configuration.
    2. ETL (Data Processing).
    3. Feature Engineering.
    4. Model Training & Evaluation.
    """
    
    # =================================================================
    # 1. SETUP PATHS & CONFIG
    # =================================================================
    project_root = Path(__file__).parent
    config_path = project_root / "src" / "noshow_lib" / "config.yaml"
    
    logger.info(f"Loading configuration from: {config_path}")
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.critical(f"Failed to load config: {e}")
        return

    # Extract paths from YAML
    raw_rel_path = config["data"]["raw_data_path"]
    processed_rel_path = config["data"]["processed_data_path"]
    
    raw_path = project_root / raw_rel_path
    processed_path = project_root / processed_rel_path

    # Define Output Paths
    features_path = project_root / "data" / "03 - features" / "dataset_final.csv"
    
    # Extract output paths from config (added for consistency)
    output_cfg = config.get("output", {})
    model_path = project_root / output_cfg.get("model_path", "models/pipeline.joblib")
    metrics_path = project_root / output_cfg.get("metrics_path", "models/metrics.csv")

    # =================================================================
    # 2. DATA PROCESSING (Clean & Filter)
    # =================================================================
    logger.info("[Stage 1/3] Starting Data Processing...")
    
    try:
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
    logger.info("[Stage 2/3] Starting Feature Engineering...")
    
    try:
        df_features = build_features(
            df=df_clean, 
            data_cfg=config.get("data")
        )
        
        # Save the feature dataset physically so the training module can read it
        features_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(features_path, index=False)
        logger.info(f"Feature dataset saved at: {features_path}")
        
    except Exception as e:
        logger.critical(f"Feature Engineering failed: {e}")
        return

    # =================================================================
    # 4. MODEL TRAINING & EVALUATION
    # =================================================================
    logger.info("[Stage 3/3] Starting Model Training...")
    
    try:
        train_model(
            config=config,                   # Config required for local mode
            data_path=features_path,         # Input: The CSV generated in step 3
            model_output_path=model_path,    # Output: The trained model
            metrics_output_path=metrics_path, # Output: The metrics CSV
            is_external_access=False         # Local mode: save to disk
        )
        
        logger.info("Pipeline finished successfully.")
        
    except Exception as e:
        logger.critical(f"Model Training failed: {e}")
        return

if __name__ == "__main__":
    main()