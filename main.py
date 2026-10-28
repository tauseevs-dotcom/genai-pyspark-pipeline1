"""Main Orchestration Script for E-Commerce Data Pipeline.

This script executes the synthetic data generation pipeline:
1. Loads settings from `src.config`.
2. Initializes `SyntheticDataGenerator`.
3. Synthesizes customers, products, and orders datasets.
4. Serializes datasets into Apache Parquet format in `data/raw/`.
5. Logs and prints detailed generation benchmarks (execution time and file sizes).
6. Implements robust exception handling and graceful error reporting.
"""

from datetime import datetime
import logging
from pathlib import Path
import sys
import time
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
from pyspark.sql import DataFrame

from src.config import (
    DATA_GEN_CONFIG,
    RAW_DATA_DIR,
    DataGenConfig,
    setup_logging,
)
from src.data_generator import SyntheticDataGenerator
from src.spark_analytics import SalesAnalytics

# Configure structured application logger
logger: logging.Logger = setup_logging(name="main_pipeline")


def format_file_size(size_bytes: int) -> str:
    """Format byte size into human-readable units (B, KB, MB, GB).

    Args:
        size_bytes: Size in integer bytes.

    Returns:
        Formatted string representing size (e.g. '14.25 MB').
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


def run_data_pipeline(
    config: DataGenConfig = DATA_GEN_CONFIG,
    output_dir: Path = RAW_DATA_DIR,
) -> Dict[str, Tuple[int, float, int]]:
    """Execute end-to-end data generation, Parquet serialization, and benchmarking.

    Args:
        config: Data generation configuration parameters.
        output_dir: Directory where raw Parquet files are saved.

    Returns:
        Dictionary mapping table name to a tuple of (row_count, elapsed_seconds, file_size_bytes).

    Raises:
        Exception: Propagates any generation or serialization error for top-level handling.
    """
    logger.info("=" * 70)
    logger.info("STARTING E-COMMERCE SYNTHETIC DATA GENERATION PIPELINE")
    logger.info("=" * 70)
    logger.info(
        "Target Scale: Customers=%d, Products=%d, Orders=%d | Seed=%d",
        config.num_customers,
        config.num_products,
        config.num_orders,
        config.random_seed,
    )
    logger.info("Output Storage: %s", output_dir.resolve())

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = SyntheticDataGenerator(
        num_customers=config.num_customers,
        num_products=config.num_products,
        num_orders=config.num_orders,
        seed=config.random_seed,
    )

    metrics: Dict[str, Tuple[int, float, int]] = {}
    overall_start_time: float = time.time()

    # --------------------------------------------------------------------------
    # 1. Generate & Save Customers
    # --------------------------------------------------------------------------
    logger.info(">>> STEP 1/3: Generating Customers...")
    t0 = time.time()
    customers_df: pd.DataFrame = generator.generate_customers()
    t_cust = time.time() - t0

    cust_parquet_path = output_dir / "customers.parquet"
    logger.info("Writing customers to Parquet: %s", cust_parquet_path)
    customers_df.to_parquet(cust_parquet_path, engine="pyarrow", index=False)
    cust_file_size = cust_parquet_path.stat().st_size
    metrics["customers"] = (len(customers_df), t_cust, cust_file_size)
    logger.info(
        "Customers completed: %d rows | Time: %.2fs | Size: %s",
        len(customers_df),
        t_cust,
        format_file_size(cust_file_size),
    )

    # --------------------------------------------------------------------------
    # 2. Generate & Save Products
    # --------------------------------------------------------------------------
    logger.info(">>> STEP 2/3: Generating Products Catalog...")
    t0 = time.time()
    products_df: pd.DataFrame = generator.generate_products()
    t_prod = time.time() - t0

    prod_parquet_path = output_dir / "products.parquet"
    logger.info("Writing products to Parquet: %s", prod_parquet_path)
    products_df.to_parquet(prod_parquet_path, engine="pyarrow", index=False)
    prod_file_size = prod_parquet_path.stat().st_size
    metrics["products"] = (len(products_df), t_prod, prod_file_size)
    logger.info(
        "Products completed: %d rows | Time: %.2fs | Size: %s",
        len(products_df),
        t_prod,
        format_file_size(prod_file_size),
    )

    # --------------------------------------------------------------------------
    # 3. Generate & Save Orders (Pareto 80/20 Skew)
    # --------------------------------------------------------------------------
    logger.info(">>> STEP 3/3: Generating Orders (Pareto 80/20 Distribution)...")
    t0 = time.time()
    orders_df: pd.DataFrame = generator.generate_orders(
        customers_df=customers_df,
        products_df=products_df,
    )
    t_ord = time.time() - t0

    ord_parquet_path = output_dir / "orders.parquet"
    logger.info("Writing orders to Parquet: %s", ord_parquet_path)
    orders_df.to_parquet(ord_parquet_path, engine="pyarrow", index=False)
    ord_file_size = ord_parquet_path.stat().st_size
    metrics["orders"] = (len(orders_df), t_ord, ord_file_size)
    logger.info(
        "Orders completed: %d rows | Time: %.2fs | Size: %s",
        len(orders_df),
        t_ord,
        format_file_size(ord_file_size),
    )

    total_time = time.time() - overall_start_time
    total_size = sum(m[2] for m in metrics.values())

    # --------------------------------------------------------------------------
    # Summary Report Display
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("           E-COMMERCE SYNTHETIC DATA GENERATION SUMMARY REPORT          ")
    print("=" * 80)
    print(f"{'Dataset':<16} | {'Rows':<12} | {'Time (sec)':<12} | {'File Size':<12} | {'Format'}")
    print("-" * 80)
    for name, (rows, elapsed, fsize) in metrics.items():
        print(
            f"{name.capitalize():<16} | {rows:<12,d} | {elapsed:<12.2f} | "
            f"{format_file_size(fsize):<12} | Parquet"
        )
    print("-" * 80)
    print(
        f"{'TOTAL':<16} | {sum(m[0] for m in metrics.values()):<12,d} | "
        f"{total_time:<12.2f} | {format_file_size(total_size):<12} | 3 Files"
    )
    print("=" * 80)
    print(f"Target Directory: {output_dir.resolve()}\n")

    return metrics


def run_sales_analytics(raw_data_dir: Path = RAW_DATA_DIR) -> None:
    """Load generated Parquet datasets and display sales analytics results.

    Args:
        raw_data_dir: Directory containing the customers, products, and orders
            Parquet datasets.
    """
    spark = SalesAnalytics.create_spark_session()
    try:
        analytics = SalesAnalytics(spark)

        datasets: Dict[str, DataFrame] = {}
        for name in ("customers", "products", "orders"):
            start_time = time.perf_counter()
            datasets[name] = analytics.load_parquet(raw_data_dir / f"{name}.parquet")
            print(f"Loaded {name}.parquet in {time.perf_counter() - start_time:.2f} seconds")

        customers_df = datasets["customers"]
        products_df = datasets["products"]
        orders_df = datasets["orders"]
        logger.info("Loaded customers dataset with columns: %s", customers_df.columns)

        operations: Tuple[Tuple[str, Callable[[], DataFrame]], ...] = (
            (
                "Top customers by revenue",
                lambda: analytics.top_customers_by_revenue(orders_df, products_df),
            ),
            (
                "Sales by category",
                lambda: analytics.sales_by_category(orders_df, products_df),
            ),
            (
                "Monthly revenue trends",
                lambda: analytics.monthly_trends(orders_df, products_df),
            ),
        )
        for name, operation in operations:
            print(f"\n{name}:")
            start_time = time.perf_counter()
            result_df = operation()
            result_df.show(truncate=False)
            print(f"{name} completed in {time.perf_counter() - start_time:.2f} seconds")
    finally:
        spark.stop()


def main() -> None:
    """Application CLI entrypoint with robust exception handling."""
    try:
        run_data_pipeline(config=DATA_GEN_CONFIG, output_dir=RAW_DATA_DIR)
        run_sales_analytics(raw_data_dir=RAW_DATA_DIR)
        logger.info("Pipeline execution completed successfully.")
    except KeyboardInterrupt:
        logger.warning("\nPipeline execution cancelled by user (KeyboardInterrupt).")
        sys.exit(130)
    except FileNotFoundError as fnf_err:
        logger.critical("File system or path error occurred: %s", fnf_err, exc_info=True)
        print(f"\n[ERROR] File system error: {fnf_err}", file=sys.stderr)
        sys.exit(1)
    except MemoryError:
        logger.critical(
            "Out of Memory during data generation. Consider reducing dataset sizes in config.py.",
            exc_info=True,
        )
        print(
            "\n[ERROR] Out of Memory! Please reduce record counts in src/config.py.",
            file=sys.stderr,
        )
        sys.exit(2)
    except Exception as exc:
        logger.critical("Unexpected pipeline failure: %s", exc, exc_info=True)
        print(f"\n[FATAL ERROR] Pipeline failed with error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
