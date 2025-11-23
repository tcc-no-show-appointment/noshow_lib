import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Union, Any

# Scikit-learn & Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    recall_score, precision_score, accuracy_score, precision_recall_curve
)
from category_encoders import TargetEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from pandas.errors import EmptyDataError, ParserError

# Optional LightGBM support
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

logger = logging.getLogger(__name__)

def get_model_instance(model_config: Dict) -> Any:
    """
    Factory method to instantiate a model class based on the YAML configuration.

    Args:
        model_config (Dict): Dictionary containing 'type' (str) and 'parameters' (dict).

    Returns:
        Any: An instantiated scikit-learn compatible classifier.

    Raises:
        ValueError: If the model type is not supported.
        ImportError: If LGBMClassifier is requested but not installed.
        Exception: If invalid parameters are passed to the model constructor.
    """
    model_type = model_config.get("type")
    params = model_config.get("parameters", {})

    logger.info(f"Instantiating model of type: {model_type}")

    try:
        if model_type == "RandomForest":
            return RandomForestClassifier(**params)
        
        elif model_type == "LGBMClassifier":
            if LGBMClassifier is None:
                raise ImportError("LightGBM is not installed. Please install it to use LGBMClassifier.")
            return LGBMClassifier(**params)
        
        else:
            raise ValueError(f"Model '{model_type}' is not supported or not implemented.")
            
    except TypeError as e:
        logger.error(f"Invalid parameters provided for model {model_type}: {e}")
        raise

def optimize_threshold(y_true: np.ndarray, y_probs: np.ndarray, metric_primary: str = "f1") -> float:
    """
    Finds the optimal decision threshold to maximize a specific metric.
    Useful for imbalanced datasets where the default 0.5 threshold is suboptimal.

    Args:
        y_true (np.array): True binary labels.
        y_probs (np.array): Predicted probabilities for the positive class.
        metric_primary (str): Metric to maximize ('f1', 'recall', 'precision').

    Returns:
        float: The optimal threshold value.
    """
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        
        # Handle edge case with empty arrays
        if len(thresholds) == 0:
            logger.warning("Empty threshold array generated. Defaulting to 0.5")
            return 0.5

        # Calculate F1 for every threshold
        # Adding epsilon to avoid division by zero
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # precision_recall_curve returns one more value for prec/recall than thresholds
        f1_scores = f1_scores[:-1]
        recall = recall[:-1]
        precision = precision[:-1]

        if metric_primary == "recall":
            # Argmax finds the index of the maximum value
            best_idx = np.argmax(recall)
        elif metric_primary == "precision":
            best_idx = np.argmax(precision)
        else:
            # Default: Maximize F1
            best_idx = np.argmax(f1_scores)
            
        best_thresh = thresholds[best_idx]
        logger.info(f"Threshold optimized for '{metric_primary}': {best_thresh:.4f}")
        return float(best_thresh)

    except Exception as e:
        logger.error(f"Error during threshold optimization: {e}. Defaulting to 0.5")
        return 0.5

def temporal_split(df: pd.DataFrame, target_col: str, date_col: str, test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into training and testing sets based on a chronological order.

    Args:
        df (pd.DataFrame): The input dataframe.
        target_col (str): The name of the target variable column.
        date_col (str): The name of the datetime column to sort by.
        test_frac (float): The fraction of data to use for testing (0.0 to 1.0).

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)

    Raises:
        KeyError: If target or date columns are missing.
        ValueError: If datetime conversion fails.
    """
    # Validation
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    if date_col not in df.columns:
        raise KeyError(f"Date column '{date_col}' not found in DataFrame.")

    try:
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Drop rows without date and sort
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        
        # Calculate split index
        split_idx = int(len(df) * (1 - test_frac))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        return (
            train_df.drop(columns=[target_col]), 
            test_df.drop(columns=[target_col]), 
            train_df[target_col], 
            test_df[target_col]
        )
    except Exception as e:
        logger.error(f"Failed during temporal split: {e}")
        raise

def train_model(
    df_input: pd.DataFrame = None,
    is_external_access: bool = False,
    config: Dict = None,
    data_path: Union[str, Path] = None, 
    model_output_path: Union[str, Path] = None, 
    metrics_output_path: Union[str, Path] = None
) -> Union[None, Tuple[bytes, Dict]]:
    """
    Orchestrates the model training pipeline: loading, splitting, preprocessing, 
    training, evaluating, and saving artifacts based on YAML configuration.

    Args:
        df_input (pd.DataFrame): Input DataFrame with features. Required when is_external_access=True.
        is_external_access (bool): If True, uses df_input, loads config from local config.yaml, 
                                   and returns (model_bytes, metrics_dict) instead of saving to disk.
        config (Dict): Parsed YAML configuration dictionary. Required when is_external_access=False.
        data_path (str | Path): Path to the feature-engineered CSV. Required when is_external_access=False.
        model_output_path (str | Path): Path to save the trained .joblib pipeline. Required when is_external_access=False.
        metrics_output_path (str | Path): Path to save evaluation metrics CSV. Required when is_external_access=False.

    Returns:
        None: When is_external_access=False (local mode).
        Tuple[bytes, Dict]: When is_external_access=True, returns (model_joblib_bytes, metrics_dict).

    Raises:
        FileNotFoundError: If data_path does not exist (local mode) or config.yaml not found (external mode).
        ValueError: If required parameters are missing for the selected mode.
        KeyError: If required configuration keys are missing.
        RuntimeError: If training fails.
    """
    # =========================================================
    # 1. Setup Paths, Load Config and Data
    # =========================================================
    
    # External Access Mode: use DataFrame input, load config from local file
    if is_external_access:
        if df_input is None:
            raise ValueError("df_input must be provided when is_external_access=True")
        
        logger.info("External access mode: using DataFrame input")
        df = df_input.copy()
        
        # Load config from local config.yaml
        try:
            from noshow_lib.config import load_config
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"config.yaml not found at: {config_path}")
            config = load_config(config_path)
            logger.info(f"Loaded configuration from: {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config in external mode: {e}")
            raise
    
    # Local Mode: use provided paths and config for loading and saving
    else:
        if config is None:
            raise ValueError("config must be provided when is_external_access=False")
        if data_path is None:
            raise ValueError("data_path must be provided when is_external_access=False")
        if model_output_path is None or metrics_output_path is None:
            raise ValueError("model_output_path and metrics_output_path must be provided when is_external_access=False")
        
        data_path = Path(data_path)
        model_output_path = Path(model_output_path)
        metrics_output_path = Path(metrics_output_path)
        
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        try:
            logger.info(f"Loading training data from: {data_path}")
            df = pd.read_csv(data_path)
        except (EmptyDataError, ParserError) as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}")
            raise

    # =========================================================
    # 2. Parse Configuration
    # =========================================================
    try:
        target_col = config["data"]["target_column"]
        split_cfg = config.get("split", {})
        model_cfg = config.get("model", {})
        eval_cfg = config.get("evaluation", {})
        balancing_cfg = config.get("balancing", {})
    except KeyError as e:
        logger.error(f"Missing required key in configuration: {e}")
        raise

    # =========================================================
    # 3. Data Splitting
    # =========================================================
    logger.info("Splitting data (Temporal Strategy)...")
    X_train, X_test, y_train, y_test = temporal_split(
        df, target_col, 
        date_col=split_cfg.get("date_column", "AppointmentDay"), 
        test_frac=split_cfg.get("test_frac", 0.2)
    )

    # Drop datetime columns from features (models generally can't handle raw dates)
    cols_date = X_train.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns
    if len(cols_date) > 0:
        logger.info(f"Dropping datetime columns from training features: {cols_date.tolist()}")
        X_train = X_train.drop(columns=cols_date)
        X_test = X_test.drop(columns=cols_date)

    # =========================================================
    # 4. Preprocessing Pipeline Construction
    # =========================================================
    try:
        num_features = X_train.select_dtypes(include=np.number).columns.tolist()
        cat_features = X_train.select_dtypes(exclude=['number', 'datetime']).columns.tolist()
        
        # Determine low/high cardinality for categorical encoding
        cat_low = [c for c in cat_features if X_train[c].nunique() < 10]
        cat_high = [c for c in cat_features if X_train[c].nunique() >= 10]

        logger.info(f"Features detected - Numeric: {len(num_features)}, Cat(Low): {len(cat_low)}, Cat(High): {len(cat_high)}")

        transformers = [
            ("num", StandardScaler(), num_features),
            ("cat_low", OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_low),
            ("cat_high", TargetEncoder(handle_unknown='value'), cat_high)
        ]
        # remainder='drop' ensures we only use selected features
        preprocessor = ColumnTransformer(transformers, remainder="drop")

        # =========================================================
        # 5. Model Instantiation
        # =========================================================
        clf = get_model_instance(model_cfg)

        # =========================================================
        # 6. Assemble Imbalanced Pipeline
        # =========================================================
        steps = [('preprocess', preprocessor)]
        
        if balancing_cfg.get("enabled"):
            smote_strategy = balancing_cfg.get("smote", {}).get("sampling_strategy", 0.5)
            logger.info(f"Applying Class Balancing (SMOTE) with strategy: {smote_strategy}")
            
            steps.append(('sampler', SMOTE(
                sampling_strategy=smote_strategy,
                random_state=42
            )))
        
        steps.append(('clf', clf))
        pipeline = ImbPipeline(steps)

        # =========================================================
        # 7. Training
        # =========================================================
        logger.info("Starting model training...")
        pipeline.fit(X_train, y_train)

        # =========================================================
        # 8. Evaluation & Optimization
        # =========================================================
        logger.info("Calculating predictions on test set...")
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Threshold Optimization Logic
        threshold = 0.5
        thresh_opt = eval_cfg.get("threshold_optimization", {})
        
        if thresh_opt.get("enabled"):
            logger.info("Optimizing decision threshold...")
            metric_to_opt = thresh_opt.get("optimize_for", {}).get("primary", "f1")
            threshold = optimize_threshold(y_test, y_prob, metric_primary=metric_to_opt)
        
        logger.info(f"Final threshold applied: {threshold:.4f}")
        y_pred = (y_prob >= threshold).astype(int)

        # =========================================================
        # 9. Metrics Calculation
        # =========================================================
        metrics_to_calc = eval_cfg.get("metrics", ["accuracy"])
        metrics_results = {"threshold_used": threshold}

        if "roc_auc" in metrics_to_calc:
            metrics_results["roc_auc"] = roc_auc_score(y_test, y_prob)
        if "average_precision" in metrics_to_calc:
            metrics_results["average_precision"] = average_precision_score(y_test, y_prob)
        if "f1" in metrics_to_calc:
            metrics_results["f1"] = f1_score(y_test, y_pred)
        if "recall" in metrics_to_calc:
            metrics_results["recall"] = recall_score(y_test, y_pred)
        if "precision" in metrics_to_calc:
            metrics_results["precision"] = precision_score(y_test, y_pred)
        if "accuracy" in metrics_to_calc:
            metrics_results["accuracy"] = accuracy_score(y_test, y_pred)

        logger.info(f"Evaluation Results: {metrics_results}")

        # =========================================================
        # 10. Save Artifacts or Return Results
        # =========================================================
        
        if is_external_access:
            # External mode: serialize model to bytes and return with metrics
            logger.info("External access mode: serializing model to bytes")
            from io import BytesIO
            buffer = BytesIO()
            joblib.dump(pipeline, buffer)
            model_bytes = buffer.getvalue()
            
            logger.info(f"Model serialized: {len(model_bytes)} bytes")
            return (model_bytes, metrics_results)
        
        else:
            # Local mode: save to disk
            # Save Model
            try:
                model_output_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(pipeline, model_output_path)
                logger.info(f"Model saved successfully to: {model_output_path}")
            except PermissionError:
                logger.error(f"Permission denied saving model to: {model_output_path}")
                raise
            
            # Save Metrics
            try:
                metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([metrics_results]).to_csv(metrics_output_path, index=False)
                logger.info(f"Metrics saved successfully to: {metrics_output_path}")
            except PermissionError:
                logger.error(f"Permission denied saving metrics to: {metrics_output_path}")
                raise

    except Exception as e:
        logger.error(f"Critical failure during training pipeline: {e}")
        raise