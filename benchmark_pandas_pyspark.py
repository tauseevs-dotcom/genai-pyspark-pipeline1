"""Compare Pandas and PySpark sales aggregation performance on Parquet data."""

from pathlib import Path
import time
from typing import Callable, List, Tuple, TypeVar

import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import RAW_DATA_DIR
from src.spark_analytics import SalesAnalytics

T = TypeVar("T")


def time_operation(operation: Callable[[], T]) -> Tuple[T, float]:
    """Run an operation and return its result and elapsed wall-clock seconds."""
    start_time = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - start_time


def run_benchmark(raw_data_dir: Path = RAW_DATA_DIR) -> None:
    """Benchmark loading, joining, aggregation, and top-10 selection.

    Spark DataFrames are persisted and counted after each lazy transformation so
    each stage's timing includes its execution rather than only query planning.

    Args:
        raw_data_dir: Directory containing ``orders.parquet`` and
            ``products.parquet``.
    """
    spark: SparkSession = SalesAnalytics.create_spark_session()
    cached_frames: List[DataFrame] = []

    try:
        orders_path = raw_data_dir / "orders.parquet"
        products_path = raw_data_dir / "products.parquet"

        (orders_pd, products_pd), pandas_load_time = time_operation(
            lambda: (
                pd.read_parquet(orders_path),
                pd.read_parquet(products_path),
            )
        )

        def load_spark_data() -> Tuple[DataFrame, DataFrame]:
            orders_df = spark.read.parquet(str(orders_path)).persist(
                StorageLevel.MEMORY_AND_DISK
            )
            products_df = spark.read.parquet(str(products_path)).persist(
                StorageLevel.MEMORY_AND_DISK
            )
            cached_frames.extend((orders_df, products_df))
            orders_df.count()
            products_df.count()
            return orders_df, products_df

        (orders_spark, products_spark), spark_load_time = time_operation(
            load_spark_data
        )
        print(
            f"Loaded {len(orders_pd):,} orders and {len(products_pd):,} products "
            f"from {raw_data_dir}"
        )

        pandas_joined, pandas_join_time = time_operation(
            lambda: orders_pd.merge(
                products_pd[["product_id", "price"]],
                on="product_id",
                how="inner",
            ).assign(revenue=lambda frame: frame["quantity"] * frame["price"])
        )

        def join_spark_data() -> DataFrame:
            joined_df = (
                orders_spark.join(products_spark, on="product_id", how="inner")
                .withColumn("revenue", F.col("quantity") * F.col("price"))
                .persist(StorageLevel.MEMORY_AND_DISK)
            )
            cached_frames.append(joined_df)
            joined_df.count()
            return joined_df

        spark_joined, spark_join_time = time_operation(join_spark_data)

        pandas_totals, pandas_aggregate_time = time_operation(
            lambda: pandas_joined.groupby("customer_id", as_index=False)["revenue"]
            .sum()
            .rename(columns={"revenue": "total_revenue"})
        )

        def aggregate_spark_data() -> DataFrame:
            totals_df = (
                spark_joined.groupBy("customer_id")
                .agg(F.sum("revenue").alias("total_revenue"))
                .persist(StorageLevel.MEMORY_AND_DISK)
            )
            cached_frames.append(totals_df)
            totals_df.count()
            return totals_df

        spark_totals, spark_aggregate_time = time_operation(aggregate_spark_data)

        pandas_top, pandas_top_time = time_operation(
            lambda: pandas_totals.sort_values(
                ["total_revenue", "customer_id"],
                ascending=[False, True],
            ).head(10)
        )
        spark_top, spark_top_time = time_operation(
            lambda: spark_totals.orderBy(
                F.col("total_revenue").desc(),
                F.col("customer_id").asc(),
            )
            .limit(10)
            .collect()
        )

        print("\nPandas top 10 customers by revenue:")
        print(pandas_top.to_string(index=False))
        print("\nPySpark top 10 customers by revenue:")
        for row in spark_top:
            print(f"{row['customer_id']}  {row['total_revenue']:.2f}")

        timings = (
            ("Pandas", pandas_load_time, pandas_join_time, pandas_aggregate_time, pandas_top_time),
            ("PySpark", spark_load_time, spark_join_time, spark_aggregate_time, spark_top_time),
        )
        print("\nPerformance comparison (seconds)")
        print(
            f"{'Framework':<10} | {'Load':>10} | {'Join + revenue':>15} | "
            f"{'Group by customer':>18} | {'Top 10':>10} | {'Total':>10}"
        )
        print("-" * 85)
        for framework, load_time, join_time, aggregate_time, top_time in timings:
            total_time = load_time + join_time + aggregate_time + top_time
            print(
                f"{framework:<10} | {load_time:>10.3f} | {join_time:>15.3f} | "
                f"{aggregate_time:>18.3f} | {top_time:>10.3f} | {total_time:>10.3f}"
            )
    finally:
        for frame in cached_frames:
            frame.unpersist(blocking=True)
        spark.stop()


if __name__ == "__main__":
    run_benchmark()
