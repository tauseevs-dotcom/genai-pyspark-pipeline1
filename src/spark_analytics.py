"""PySpark E-commerce Analytics Pipeline.

This module loads synthetic raw e-commerce data (customers, products, and orders),
executes business intelligence queries and statistical metric calculations,
and saves the analyzed results to the processed storage zone.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SPARK_CONFIG,
    SparkConfig,
    setup_logging,
)

logger: logging.Logger = setup_logging(name="spark_analytics")


def create_spark_session(config: SparkConfig = SPARK_CONFIG) -> SparkSession:
    """Create and configure a local or cluster PySpark session.

    Args:
        config: SparkConfig dataclass instance with execution settings.

    Returns:
        An active SparkSession instance.
    """
    logger.info("Initializing SparkSession: app_name=%s, master=%s", config.app_name, config.master)

    spark: SparkSession = (
        SparkSession.builder.appName(config.app_name)
        .master(config.master)
        .config("spark.sql.shuffle.partitions", str(config.shuffle_partitions))
        .config("spark.driver.memory", config.driver_memory)
        .config("spark.sql.session.timeZone", config.timezone)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ui.enabled", "false")  # Disable web UI for local batch runs
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(config.log_level)
    logger.info("SparkSession initialized successfully. PySpark version: %s", spark.version)
    return spark


def load_raw_data(
    spark: SparkSession, raw_data_dir: Path = RAW_DATA_DIR
) -> Dict[str, DataFrame]:
    """Load raw e-commerce datasets from CSV or Parquet files into Spark DataFrames.

    Args:
        spark: Active SparkSession.
        raw_data_dir: Directory containing raw input files.

    Returns:
        Dictionary containing DataFrames for 'customers', 'products', and 'orders'.

    Raises:
        FileNotFoundError: If any required dataset is missing.
    """
    logger.info("Loading raw datasets from %s...", raw_data_dir)
    datasets: Dict[str, DataFrame] = {}
    table_names = ["customers", "products", "orders"]

    for table in table_names:
        csv_file = raw_data_dir / f"{table}.csv"
        parquet_file = raw_data_dir / f"{table}.parquet"

        if parquet_file.exists():
            logger.info("Loading '%s' from Parquet: %s", table, parquet_file)
            df = spark.read.parquet(str(parquet_file))
        elif csv_file.exists():
            logger.info("Loading '%s' from CSV: %s", table, csv_file)
            df = (
                spark.read.option("header", "true")
                .option("inferSchema", "true")
                .csv(str(csv_file))
            )
        else:
            error_msg = f"Dataset file for '{table}' not found in {raw_data_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info("Loaded table '%s' with %d records.", table, df.count())
        datasets[table] = df

    return datasets


def compute_top_selling_products(
    products_df: DataFrame,
    orders_df: DataFrame,
    top_n: int = 10,
) -> DataFrame:
    """Identify top-performing products by total units sold and revenue.

    Args:
        products_df: DataFrame containing product catalog data.
        orders_df: DataFrame containing order transaction records.
        top_n: Number of top products to return.

    Returns:
        DataFrame sorted by total revenue descending.
    """
    logger.info("Calculating Top %d Selling Products...", top_n)

    joined_df = orders_df.join(products_df, on="product_id", how="inner").withColumn(
        "line_revenue", F.round(F.col("quantity") * F.col("price"), 2)
    )

    result_df = (
        joined_df.groupBy("product_id", "name", "category")
        .agg(
            F.sum("quantity").alias("total_units_sold"),
            F.round(F.sum("line_revenue"), 2).alias("total_revenue"),
            F.countDistinct("order_id").alias("order_count"),
            F.first("rating").alias("rating"),
        )
        .orderBy(F.col("total_revenue").desc())
        .limit(top_n)
    )

    return result_df


def compute_revenue_by_category(
    products_df: DataFrame,
    orders_df: DataFrame,
) -> DataFrame:
    """Analyze total sales, units sold, and average order value across product categories.

    Args:
        products_df: DataFrame containing product catalog.
        orders_df: DataFrame containing customer orders.

    Returns:
        DataFrame summarizing sales performance per category.
    """
    logger.info("Computing revenue and performance metrics by category...")

    joined_df = orders_df.join(products_df, on="product_id", how="inner").withColumn(
        "line_revenue", F.round(F.col("quantity") * F.col("price"), 2)
    )

    result_df = (
        joined_df.groupBy("category")
        .agg(
            F.countDistinct("order_id").alias("total_orders"),
            F.sum("quantity").alias("total_units_sold"),
            F.round(F.sum("line_revenue"), 2).alias("total_revenue"),
            F.round(F.avg("line_revenue"), 2).alias("avg_line_revenue"),
        )
        .orderBy(F.col("total_revenue").desc())
    )

    return result_df


def compute_customer_lifetime_value(
    customers_df: DataFrame,
    products_df: DataFrame,
    orders_df: DataFrame,
) -> DataFrame:
    """Calculate Customer Lifetime Value (CLV), total spend, and customer value tier.

    Args:
        customers_df: DataFrame containing customer profiles.
        products_df: DataFrame containing product prices.
        orders_df: DataFrame containing customer orders.

    Returns:
        DataFrame with aggregated customer metrics and spend tiers.
    """
    logger.info("Computing Customer Lifetime Value (CLV)...")

    order_spend_df = orders_df.join(products_df, on="product_id", how="inner").withColumn(
        "line_revenue", F.col("quantity") * F.col("price")
    )

    customer_agg = order_spend_df.groupBy("customer_id").agg(
        F.countDistinct("order_id").alias("lifetime_orders"),
        F.sum("quantity").alias("total_units_purchased"),
        F.round(F.sum("line_revenue"), 2).alias("lifetime_spend"),
        F.min("order_date").alias("first_order_date"),
        F.max("order_date").alias("last_order_date"),
    )

    result_df = (
        customers_df.join(customer_agg, on="customer_id", how="left")
        .fillna({"lifetime_orders": 0, "total_units_purchased": 0, "lifetime_spend": 0.0})
        .withColumn(
            "spend_tier",
            F.when(F.col("lifetime_spend") >= 2000.0, "VIP Tier")
            .when(F.col("lifetime_spend") >= 1000.0, "High Value")
            .when(F.col("lifetime_spend") >= 300.0, "Medium Value")
            .otherwise("Standard"),
        )
        .select(
            "customer_id",
            "name",
            "email",
            "age",
            "city",
            "country",
            "lifetime_orders",
            "total_units_purchased",
            "lifetime_spend",
            "first_order_date",
            "last_order_date",
            "spend_tier",
        )
        .orderBy(F.col("lifetime_spend").desc())
    )

    return result_df


def compute_monthly_sales_trend(
    products_df: DataFrame,
    orders_df: DataFrame,
) -> DataFrame:
    """Compute month-over-month sales performance and growth rates.

    Args:
        products_df: DataFrame containing product catalog data.
        orders_df: DataFrame containing orders with dates and quantities.

    Returns:
        DataFrame with monthly aggregated sales and MoM growth percentage.
    """
    logger.info("Computing monthly sales and order volume trends...")

    joined_df = (
        orders_df.join(products_df, on="product_id", how="inner")
        .withColumn("line_revenue", F.col("quantity") * F.col("price"))
        .withColumn("order_month", F.date_format(F.col("order_date").cast("date"), "yyyy-MM"))
    )

    monthly_summary = joined_df.groupBy("order_month").agg(
        F.countDistinct("order_id").alias("total_orders"),
        F.sum("quantity").alias("total_units_sold"),
        F.round(F.sum("line_revenue"), 2).alias("total_revenue"),
        F.round(F.avg("line_revenue"), 2).alias("avg_order_value"),
    )

    month_window = Window.orderBy("order_month")
    result_df = (
        monthly_summary.withColumn(
            "prev_month_revenue", F.lag("total_revenue", 1).over(month_window)
        )
        .withColumn(
            "mom_revenue_growth_pct",
            F.round(
                (
                    (F.col("total_revenue") - F.col("prev_month_revenue"))
                    / F.col("prev_month_revenue")
                )
                * 100.0,
                2,
            ),
        )
        .orderBy("order_month")
    )

    return result_df


def compute_age_demographic_insights(
    customers_df: DataFrame,
    products_df: DataFrame,
    orders_df: DataFrame,
) -> DataFrame:
    """Analyze purchasing behavior and revenue across customer age brackets.

    Args:
        customers_df: DataFrame containing customer age records.
        products_df: DataFrame containing product prices.
        orders_df: DataFrame containing transaction history.

    Returns:
        DataFrame summarizing revenue and average order volume by age bracket.
    """
    logger.info("Computing demographic and age bracket purchasing insights...")

    with_brackets = customers_df.withColumn(
        "age_group",
        F.when(F.col("age") < 25, "18-24")
        .when(F.col("age") < 35, "25-34")
        .when(F.col("age") < 45, "35-44")
        .when(F.col("age") < 55, "45-54")
        .otherwise("55+"),
    )

    joined_df = (
        orders_df.join(products_df, on="product_id", how="inner")
        .join(with_brackets, on="customer_id", how="inner")
        .withColumn("line_revenue", F.col("quantity") * F.col("price"))
    )

    result_df = (
        joined_df.groupBy("age_group")
        .agg(
            F.countDistinct("customer_id").alias("active_customers"),
            F.countDistinct("order_id").alias("total_orders"),
            F.sum("quantity").alias("units_purchased"),
            F.round(F.sum("line_revenue"), 2).alias("total_revenue"),
            F.round(F.avg("line_revenue"), 2).alias("avg_spend_per_order"),
        )
        .orderBy("age_group")
    )

    return result_df


def save_processed_data(
    results: Dict[str, DataFrame],
    output_dir: Path = PROCESSED_DATA_DIR,
    save_csv: bool = True,
    save_parquet: bool = True,
) -> None:
    """Write processed analytical DataFrames to disk in CSV and/or Parquet formats.

    Args:
        results: Dictionary mapping table names to PySpark DataFrames.
        output_dir: Output target directory.
        save_csv: Whether to save as CSV.
        save_parquet: Whether to save as Parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving analytical results to %s...", output_dir)

    for name, df in results.items():
        if save_parquet:
            parquet_path = output_dir / f"{name}.parquet"
            logger.info("Writing '%s' to %s", name, parquet_path)
            df.write.mode("overwrite").parquet(str(parquet_path))

        if save_csv:
            csv_path = output_dir / f"{name}.csv"
            logger.info("Writing '%s' to %s", name, csv_path)
            pandas_df = df.toPandas()
            pandas_df.to_csv(csv_path, index=False)

    logger.info("Successfully persisted all processed results to %s", output_dir)


def run_pipeline(
    raw_data_dir: Path = RAW_DATA_DIR,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    spark_session: Optional[SparkSession] = None,
) -> Dict[str, DataFrame]:
    """Execute end-to-end PySpark analytics pipeline.

    Args:
        raw_data_dir: Path to raw data directory.
        processed_data_dir: Path to processed output directory.
        spark_session: Optional existing SparkSession. If None, one will be created.

    Returns:
        Dictionary of computed metric DataFrames.
    """
    should_stop_spark = False
    spark = spark_session

    if spark is None:
        spark = create_spark_session()
        should_stop_spark = True

    try:
        logger.info("Starting PySpark analytics execution pipeline...")

        # 1. Load raw data
        raw_tables = load_raw_data(spark=spark, raw_data_dir=raw_data_dir)
        customers_df = raw_tables["customers"]
        products_df = raw_tables["products"]
        orders_df = raw_tables["orders"]

        # 2. Compute Analytics
        top_products_df = compute_top_selling_products(
            products_df=products_df,
            orders_df=orders_df,
            top_n=10,
        )

        category_perf_df = compute_revenue_by_category(
            products_df=products_df,
            orders_df=orders_df,
        )

        customer_ltv_df = compute_customer_lifetime_value(
            customers_df=customers_df,
            products_df=products_df,
            orders_df=orders_df,
        )

        monthly_sales_df = compute_monthly_sales_trend(
            products_df=products_df,
            orders_df=orders_df,
        )

        demographics_df = compute_age_demographic_insights(
            customers_df=customers_df,
            products_df=products_df,
            orders_df=orders_df,
        )

        analytics_results: Dict[str, DataFrame] = {
            "top_selling_products": top_products_df,
            "category_performance": category_perf_df,
            "customer_lifetime_value": customer_ltv_df,
            "monthly_sales_trend": monthly_sales_df,
            "age_demographics": demographics_df,
        }

        # 3. Log Previews
        logger.info("=== PREVIEW: Top Selling Products ===")
        top_products_df.show(5, truncate=False)

        logger.info("=== PREVIEW: Category Performance ===")
        category_perf_df.show(truncate=False)

        logger.info("=== PREVIEW: Age Demographics ===")
        demographics_df.show(truncate=False)

        # 4. Save results to data/processed/
        save_processed_data(results=analytics_results, output_dir=processed_data_dir)

        logger.info("PySpark analytics pipeline completed successfully.")
        return analytics_results

    finally:
        if should_stop_spark and spark is not None:
            logger.info("Stopping SparkSession...")
            spark.stop()
            logger.info("SparkSession stopped.")


class SalesAnalytics:
    """Provide focused sales analytics over order and product DataFrames."""

    def __init__(self, spark: Optional[SparkSession] = None) -> None:
        """Initialize analytics with an existing or newly created Spark session.

        Args:
            spark: An existing Spark session, or ``None`` to create one locally.
        """
        self.spark = spark if spark is not None else self.create_spark_session()

    @staticmethod
    def create_spark_session() -> SparkSession:
        """Create a local Spark session with 4 GB memory and performance settings.

        Returns:
            A configured active Spark session.
        """
        return create_spark_session()

    def load_parquet(self, path: Union[str, Path]) -> DataFrame:
        """Load Parquet data from a file or directory into a DataFrame.

        Args:
            path: Path to the Parquet file or directory.

        Returns:
            A DataFrame containing the loaded Parquet data.
        """
        return self.spark.read.parquet(str(path))

    def top_customers_by_revenue(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Return the customers with the highest total product revenue.

        Args:
            orders_df: Orders containing ``customer_id``, ``product_id``, and
                ``quantity`` columns.
            products_df: Products containing ``product_id`` and ``price`` columns.
            n: Number of top customers to return; must be positive.

        Returns:
            A DataFrame with ``customer_id`` and ``total_revenue``, ordered by
            revenue descending.

        Raises:
            ValueError: If ``n`` is not positive.
        """
        if n < 1:
            raise ValueError("n must be a positive integer")

        return (
            orders_df.join(products_df, on="product_id", how="inner")
            .groupBy("customer_id")
            .agg(
                F.round(
                    F.sum(F.col("quantity") * F.col("price")),
                    2,
                ).alias("total_revenue")
            )
            .orderBy(F.col("total_revenue").desc(), F.col("customer_id").asc())
            .limit(n)
        )

    def sales_by_category(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Summarize revenue and units sold for each product category.

        Args:
            orders_df: Orders containing ``product_id`` and ``quantity`` columns.
            products_df: Products containing ``product_id``, ``category``, and
                ``price`` columns.

        Returns:
            A DataFrame with category, total revenue, and total units sold.
        """
        return (
            orders_df.join(products_df, on="product_id", how="inner")
            .groupBy("category")
            .agg(
                F.round(
                    F.sum(F.col("quantity") * F.col("price")),
                    2,
                ).alias("total_revenue"),
                F.sum("quantity").alias("total_units_sold"),
            )
            .orderBy(F.col("total_revenue").desc(), F.col("category").asc())
        )

    def monthly_trends(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Calculate monthly revenue and growth from the preceding month.

        Args:
            orders_df: Orders containing ``product_id``, ``quantity``, and
                ``order_date`` columns.
            products_df: Products containing ``product_id`` and ``price`` columns.

        Returns:
            A DataFrame with monthly revenue, prior-month revenue, and the
            month-over-month revenue growth percentage. Growth is null when
            there is no prior-month revenue or it is zero.
        """
        monthly_revenue = (
            orders_df.join(products_df, on="product_id", how="inner")
            .withColumn(
                "order_month",
                F.date_format(F.col("order_date").cast("date"), "yyyy-MM"),
            )
            .groupBy("order_month")
            .agg(
                F.round(
                    F.sum(F.col("quantity") * F.col("price")),
                    2,
                ).alias("total_revenue")
            )
        )
        month_window = Window.orderBy("order_month")
        with_previous_month = monthly_revenue.withColumn(
            "prev_month_revenue",
            F.lag("total_revenue").over(month_window),
        )

        return (
            with_previous_month.withColumn(
                "mom_revenue_growth_pct",
                F.when(
                    F.col("prev_month_revenue").isNotNull()
                    & (F.col("prev_month_revenue") != 0),
                    F.round(
                        (
                            (F.col("total_revenue") - F.col("prev_month_revenue"))
                            / F.col("prev_month_revenue")
                        )
                        * 100.0,
                        2,
                    ),
                ),
            )
            .orderBy("order_month")
        )


def main() -> None:
    """CLI entrypoint for standalone pipeline execution."""
    logger.info("Executing spark analytics script directly...")
    run_pipeline()


if __name__ == "__main__":
    main()
