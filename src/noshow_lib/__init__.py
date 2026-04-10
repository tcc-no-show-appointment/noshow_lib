__version__ = "0.3.7"

from .config import load_config
from .data_handler import load_and_validate
from .feature_engineering import build_features
from .feature_engineering_duckdb import build_features_from_parquet
from .model_training import train_model
from .model_inference import predict, load_models
from .stats_precompute import precompute_patient_stats, precompute_contextual_stats
from .stats_enrichment import enrich_with_precomputed_stats
from .logger import setup_logger

__all__ = [
    "load_config",
    "load_and_validate",
    "build_features",
    "build_features_from_parquet",
    "train_model",
    "predict",
    "load_models",
    "precompute_patient_stats",
    "precompute_contextual_stats",
    "enrich_with_precomputed_stats",
    "setup_logger",
]
