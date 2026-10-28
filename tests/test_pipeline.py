"""Unit and Integration Tests for SyntheticDataGenerator and Pipeline.

This module validates data generation integrity, schema conformity,
distribution characteristics (Normal age, Pareto orders), and relational integrity.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import pytest

from src.data_generator import SyntheticDataGenerator


@pytest.fixture
def generator() -> SyntheticDataGenerator:
    """Fixture providing initialized generator instance with small scale for fast tests.

    Returns:
        SyntheticDataGenerator instance.
    """
    return SyntheticDataGenerator(
        num_customers=200,
        num_products=50,
        num_orders=1_000,
        seed=42,
    )


def test_config_paths() -> None:
    """Verify that configuration paths are properly defined."""
    from src.config import BASE_DIR, DATA_DIR, LOGS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

    assert BASE_DIR.exists(), "BASE_DIR must exist"
    assert DATA_DIR.exists(), "DATA_DIR must exist"
    assert RAW_DATA_DIR.exists(), "RAW_DATA_DIR must exist"
    assert PROCESSED_DATA_DIR.exists(), "PROCESSED_DATA_DIR must exist"
    assert LOGS_DIR.exists(), "LOGS_DIR must exist"


def test_customer_generation(generator: SyntheticDataGenerator) -> None:
    """Validate customer generation schema, age distribution, and uniqueness."""
    customers_df: pd.DataFrame = generator.generate_customers(count=200)

    assert len(customers_df) == 200
    expected_cols = {
        "customer_id",
        "name",
        "email",
        "age",
        "city",
        "country",
        "registration_date",
    }
    assert expected_cols.issubset(set(customers_df.columns))
    assert customers_df["customer_id"].is_unique

    # Age: Normal distribution around 35
    assert (customers_df["age"] >= 18).all()
    assert (customers_df["age"] <= 80).all()
    mean_age = customers_df["age"].mean()
    assert 30 <= mean_age <= 40, f"Expected mean age near 35, got {mean_age}"


def test_product_generation(generator: SyntheticDataGenerator) -> None:
    """Validate product catalog generation, pricing, categories, stock, and ratings."""
    products_df: pd.DataFrame = generator.generate_products(count=50)

    assert len(products_df) == 50
    expected_cols = {"product_id", "name", "category", "price", "stock", "rating"}
    assert expected_cols.issubset(set(products_df.columns))

    expected_categories = {"Electronics", "Clothing", "Home", "Sports", "Books"}
    assert set(products_df["category"]).issubset(expected_categories)

    # Price bounds (10 to 500)
    assert (products_df["price"] >= 10.0).all()
    assert (products_df["price"] <= 500.0).all()

    # Stock bounds (0 to 1000)
    assert (products_df["stock"] >= 0).all()
    assert (products_df["stock"] <= 1000).all()

    # Rating bounds (1 to 5)
    assert (products_df["rating"] >= 1.0).all()
    assert (products_df["rating"] <= 5.0).all()


def test_orders_generation_and_pareto(generator: SyntheticDataGenerator) -> None:
    """Verify order schema, relational integrity, quantity bounds, and Pareto skew."""
    customers_df = generator.generate_customers(count=200)
    products_df = generator.generate_products(count=50)
    orders_df = generator.generate_orders(
        count=1_000,
        customers_df=customers_df,
        products_df=products_df,
    )

    assert len(orders_df) == 1_000
    expected_cols = {"order_id", "customer_id", "product_id", "quantity", "order_date"}
    assert expected_cols.issubset(set(orders_df.columns))

    # Quantity between 1 and 10
    assert (orders_df["quantity"] >= 1).all()
    assert (orders_df["quantity"] <= 10).all()

    # Foreign key checks
    valid_customers = set(customers_df["customer_id"])
    valid_products = set(products_df["product_id"])
    assert set(orders_df["customer_id"]).issubset(valid_customers)
    assert set(orders_df["product_id"]).issubset(valid_products)

    # Pareto principle verification: Top 20% of customers make disproportionate share of orders
    order_counts_per_cust = orders_df["customer_id"].value_counts()
    top_20_pct_count = max(1, int(len(customers_df) * 0.20))
    top_customers_orders = order_counts_per_cust.head(top_20_pct_count).sum()
    share_of_orders = top_customers_orders / len(orders_df)

    # With Pareto distribution alpha=1.161, top 20% should account for significantly > 50%
    assert share_of_orders >= 0.50, (
        f"Expected top 20% to account for majority of orders, got {share_of_orders:.2%}"
    )


def test_save_to_disk(tmp_path: Path, generator: SyntheticDataGenerator) -> None:
    """Verify saving generated datasets to disk."""
    generator.num_customers = 50
    generator.num_products = 20
    generator.num_orders = 100

    saved_paths = generator.save_to_disk(output_dir=tmp_path, save_format="csv")

    for key in ["customers_csv", "products_csv", "orders_csv"]:
        assert key in saved_paths
        file_path = saved_paths[key]
        assert file_path.exists()
        loaded_df = pd.read_csv(file_path)
        assert len(loaded_df) > 0


def test_main_pipeline_run(tmp_path: Path) -> None:
    """Verify that run_data_pipeline runs and saves Parquet files with benchmark metrics."""
    from main import format_file_size, run_data_pipeline
    from src.config import DataGenConfig

    assert format_file_size(500) == "500 B"
    assert format_file_size(1024 * 50) == "50.00 KB"
    assert format_file_size(1024 * 1024 * 5) == "5.00 MB"

    test_config = DataGenConfig(
        num_customers=30,
        num_products=15,
        num_orders=50,
        random_seed=999,
    )

    metrics = run_data_pipeline(config=test_config, output_dir=tmp_path)

    for name in ["customers", "products", "orders"]:
        assert name in metrics
        rows, elapsed, fsize = metrics[name]
        assert rows > 0
        assert elapsed >= 0
        assert fsize > 0
        parquet_file = tmp_path / f"{name}.parquet"
        assert parquet_file.exists()
        loaded_df = pd.read_parquet(parquet_file)
        assert len(loaded_df) == rows


def test_benchmark_dataframe_and_operations(tmp_path: Path) -> None:
    """Validate benchmark DataFrame generation and single format benchmark on small scale."""
    from benchmark_formats import (
        generate_benchmark_dataframe,
        benchmark_single_format,
        measure_operation,
    )

    df = generate_benchmark_dataframe(num_rows=200, seed=123)
    assert len(df) == 200
    assert list(df.columns) == ["id", "name", "email", "amount", "date", "category"]

    res = benchmark_single_format(
        format_name="Test Parquet",
        extension=".parquet",
        write_fn=lambda d, p: d.to_parquet(p, index=False),
        read_fn=lambda p: pd.read_parquet(p),
        df=df,
        target_dir=tmp_path,
    )

    assert res.file_size_mb > 0
    assert res.write_wall_time >= 0
    assert res.read_wall_time >= 0
    assert res.energy_consumption_wh >= 0
    assert res.peak_memory_mb >= 0
