# src/noshow_lib/inference.py

import pandas as pd
import joblib
from pathlib import Path
from typing import Union, Dict, Optional
import logging

from pandas.errors import EmptyDataError, ParserError

logger = logging.getLogger(__name__)

def predict(
    input_data: Union[str, Path, pd.DataFrame], 
    model_path: Union[str, Path], 
    output_path: Optional[Union[str, Path]] = None, 
    config: Optional[Dict] = None,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Runs inference using a trained model/pipeline (.joblib).
    Automatically removes the target column defined in the YAML config to prevent data leakage.

    Args:
        input_data: Path to the input CSV file OR a pandas DataFrame.
        model_path: Path to the saved pipeline (.joblib).
        output_path: (Optional) Path where the prediction CSV will be saved.
        config: Configuration dictionary (loaded from YAML) to identify target columns.
        threshold: Probability threshold for classification (default 0.5).

    Returns:
        pd.DataFrame: DataFrame containing 'probability' and 'prediction' (class).

    Raises:
        FileNotFoundError: If input data or model file does not exist.
        ValueError: If input data columns do not match what the model expects.
        RuntimeError: For issues loading the model or running predictions.
    """

    model_path = Path(model_path)
    if output_path:
        output_path = Path(output_path)

    # =========================================================
    # 1. Load Data
    # =========================================================
    try:
        if isinstance(input_data, pd.DataFrame):
            logger.info("Input data received as DataFrame.")
            df = input_data.copy()
        else:
            data_path = Path(input_data)
            if not data_path.exists():
                logger.error(f"Data file not found: {data_path}")
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            logger.info(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)

    except EmptyDataError:
        logger.error("The input CSV file is empty.")
        raise
    except ParserError:
        logger.error("Failed to parse input CSV. Check file format.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise

    # =========================================================
    # 2. Remove Target (Prevent Data Leakage)
    # =========================================================
    # Priority 1: Use column defined in Config
    target_col = None
    if config:
        target_col = config.get("data", {}).get("target_column")

    if target_col:
        if target_col in df.columns:
            logger.info(f"Removing target column '{target_col}' (from config) to prevent leakage.")
            df = df.drop(columns=[target_col])
    else:
        # Priority 2: Fallback list if no config is provided
        candidates = ["No-show", "no_show", "target", "label"]
        cols_to_drop = [col for col in candidates if col in df.columns]
        if cols_to_drop:
            logger.info(f"Removing suspected target columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

    X_predict = df

    # =========================================================
    # 3. Load Model
    # =========================================================
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        logger.info(f"Loading model pipeline from: {model_path}")
        pipeline = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model .joblib file: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

    # =========================================================
    # 4. Execute Prediction
    # =========================================================
    if not hasattr(pipeline, "predict_proba"):
        raise AttributeError("The loaded model does not have a 'predict_proba' method.")

    logger.info("Calculating probabilities...")
    
    try:
        # Extract probabilities for the positive class (index 1)
        predictions_prob = pipeline.predict_proba(X_predict)[:, 1]
    
    except ValueError as e:
        # This catches feature mismatch (e.g., model expects 10 cols, got 9)
        logger.error(f"Feature mismatch error. Ensure input data matches training features. Details: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during prediction execution: {e}")
        raise RuntimeError(f"Error during prediction: {e}")

    # Apply threshold
    predictions_label = (predictions_prob >= threshold).astype(int)

    # Create result DataFrame
    result_df = pd.DataFrame({
        "probability": predictions_prob,
        "prediction": predictions_label
    })

    # =========================================================
    # 5. Save Output
    # =========================================================
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to: {output_path}")
        except PermissionError:
            logger.error(f"Permission denied when writing to: {output_path}")
            raise
        except OSError as e:
            logger.error(f"OS error when writing output (disk full?): {e}")
            raise

    return result_df