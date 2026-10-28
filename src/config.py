"""Project Configuration Module.

This module centralizes all paths, data generation parameters, PySpark settings,
and logging configurations for the e-commerce data pipeline.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple


# ==============================================================================
# Directory and File Paths
# ==============================================================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
LOGS_DIR: Path = BASE_DIR / "logs"

# Ensure runtime directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Logging Configuration
# ==============================================================================
def setup_logging(
    name: str = "ecommerce_pipeline",
    log_level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """Set up and configure a structured logger for the application.

    Args:
        name: Name of the logger, typically the module __name__.
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_to_file: Whether to persist logs to a file in the logs directory.

    Returns:
        A configured logging.Logger instance.
    """
    logger: logging.Logger = logging.getLogger(name)

    # Avoid duplicate handlers if setup is called multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(log_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_to_file:
        log_file_path = LOGS_DIR / f"{name}.log"
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ==============================================================================
# Data Generation Configuration
# ==============================================================================
@dataclass(frozen=True)
class DataGenConfig:
    """Configuration parameters for synthetic e-commerce data generation."""

    num_customers: int = 100_000
    num_products: int = 10_000
    num_orders: int = 1_000_000
    random_seed: int = 42

    # Date range for transactions
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"

    # Category definitions with price bounds (min_price, max_price)
    categories: List[str] = field(
        default_factory=lambda: [
            "Electronics",
            "Clothing",
            "Home",
            "Sports",
            "Books",
        ]
    )

    # Customer age distribution parameters
    age_mean: float = 35.0
    age_std: float = 10.0
    age_min: int = 18
    age_max: int = 80

    # Pareto parameter alpha for 80/20 customer order skew
    pareto_alpha: float = 1.161


# ==============================================================================
# PySpark Configuration
# ==============================================================================
@dataclass(frozen=True)
class SparkConfig:
    """Configuration parameters for PySpark session and operations."""

    app_name: str = "EcommerceDataPipeline"
    master: str = "local[*]"
    log_level: str = "WARN"
    shuffle_partitions: int = 8
    driver_memory: str = "4g"
    timezone: str = "UTC"


# Default instantiated configuration objects for easy imports
DATA_GEN_CONFIG = DataGenConfig()
SPARK_CONFIG = SparkConfig()
