# src/noshow_lib/data_processing.py

import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from pathlib import Path
from typing import Union, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def load_and_process_data(
    input_data: Union[str, Path, pd.DataFrame], 
    processed_path: Optional[Union[str, Path]] = None, 
    config: Optional[Dict] = None
) -> pd.DataFrame:
    
    """
    Loads raw data (or takes a DataFrame), applies initial preprocessing based on configuration, 
    and optionally saves the processed data.

    The preprocessing steps include:
    1. Handling missing values (drop, mean, median).
    2. Renaming columns.
    3. Casting column data types.

    Args:
        input_data (str | Path | pd.DataFrame): Path to the input CSV OR a pandas DataFrame object.
        processed_path (str | Path, optional): Path where the processed CSV will be saved.
        config (dict, optional): Configuration dictionary (usually loaded from YAML).
                                Expected keys: 'handle_missing', 'rename_columns', 'cast_types'.

    Returns:
        pd.DataFrame: The processed pandas DataFrame.

    Raises:
        ValueError: If an invalid method is provided for 'handle_missing'.
        TypeError: If configuration sections are not dictionaries when expected.
        FileNotFoundError: If input_data is a path that doesn't exist.
    """
    
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raw_path = Path(input_data)
        if not raw_path.exists():
            logger.error(f"File not found: {raw_path}")
            raise FileNotFoundError(f"File not found: {raw_path}")
        
        logger.info(f"Loading raw data from: {raw_path}")
        try:
            df = pd.read_csv(raw_path)
        except EmptyDataError:
            logger.error(f"The file is empty: {raw_path}")
            raise # Interrompe o programa
        except ParserError:
            logger.error(f"The file is corrupted or not a valid CSV: {raw_path}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading CSV: {e}")
            raise

    # Early return if no config provided
    if not config:
        logger.warning("No configuration provided. Skipping preprocessing.")
    
        if processed_path:
            path_obj = Path(processed_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path_obj, index=False)
            logger.info(f"Unprocessed data saved copy to: {path_obj}")
        return df
    
    # =====================================================================
    # 1. Handle Missing Values
    # =====================================================================
    handle_missing = config.get("handle_missing")
    
    if handle_missing:
        logger.info(f"Handling missing values using method: {handle_missing}")
        if handle_missing == "drop":
            df = df.dropna()
        elif handle_missing == "mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif handle_missing == "median":
            df = df.fillna(df.median(numeric_only=True))
        elif handle_missing == "none":
            pass
        else:
            raise ValueError(f"Invalid method for handle_missing: {handle_missing}")

    # =====================================================================
    # 2. Rename Columns
    # =====================================================================
    rename_cfg = config.get("rename_columns")
    if rename_cfg:
        if isinstance(rename_cfg, dict):
            logger.info("Renaming columns...")
            df = df.rename(columns=rename_cfg)
        else:
            raise TypeError("'rename_columns' must be a dictionary.")

    # =====================================================================
    # 3. Cast Data Types
    # =====================================================================
    cast_cfg = config.get("cast_types")
    if cast_cfg:
        if isinstance(cast_cfg, dict):
            logger.info("Casting column types...")
            for col, dtype in cast_cfg.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dtype)
                    except Exception as e:
                        logger.warning(f"Could not convert column '{col}' to '{dtype}': {e}")
                else:
                    logger.warning(f"Column '{col}' not found in dataset, skipping cast.")
        else:
            raise TypeError("'cast_types' must be a dictionary.")

    # =====================================================================
    # FINAL: Save processed file
    # =====================================================================
    if processed_path:
            _save_securely(df, processed_path)

    return df


def _save_securely(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Internal helper to securely save the DataFrame to a CSV file.

    This function ensures the output directory exists before saving and 
    provides robust error logging for common I/O issues.

    Args:
        df (pd.DataFrame): The pandas DataFrame to be saved.
        path (str | Path): The destination file path.

    Raises:
        PermissionError: If the program lacks write permissions for the target directory.
        OSError: If the disk is full or another OS-level error occurs during writing.
    """
    try:
        path_obj = Path(path)
        
        # Ensure parent directory exists
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(path_obj, index=False)
        logger.info(f"Processed data saved to: {path_obj}")
        
    except PermissionError:
        logger.error(f"Permission denied when writing to: {path}")
        raise
    except OSError as e:
        logger.error(f"Disk error (full?) or OS error when writing to {path}: {e}")
        raise