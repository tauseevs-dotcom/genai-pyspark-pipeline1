"""Synthetic E-commerce Data Generator Module.

This module provides the `SyntheticDataGenerator` class to generate large-scale,
realistic e-commerce datasets (customers, products, and orders) using Faker,
NumPy statistical distributions (Normal, Pareto), and pandas DataFrames.
"""

from datetime import datetime, timedelta
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from faker import Faker
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    DATA_GEN_CONFIG,
    DataGenConfig,
    RAW_DATA_DIR,
    setup_logging,
)

logger: logging.Logger = setup_logging(name="data_generator")


class SyntheticDataGenerator:
    """High-performance synthetic e-commerce data generator.

    Generates customers, products, and orders with realistic statistical
    distributions including Normal distribution for customer age and
    Pareto (80/20 rule) distribution for order frequency.
    """

    def __init__(
        self,
        num_customers: int = 100_000,
        num_products: int = 10_000,
        num_orders: int = 1_000_000,
        seed: int = 42,
    ) -> None:
        """Initialize the synthetic data generator with target dataset sizes and random seed.

        Args:
            num_customers: Number of customer profiles to generate. Default: 100,000.
            num_products: Number of products in catalog to generate. Default: 10,000.
            num_orders: Number of purchase orders to generate. Default: 1,000,000.
            seed: Random seed for deterministic reproducibility. Default: 42.
        """
        self.num_customers: int = num_customers
        self.num_products: int = num_products
        self.num_orders: int = num_orders
        self.seed: int = seed

        # Set seeds across random, numpy, and Faker
        random.seed(self.seed)
        np.random.seed(self.seed)
        Faker.seed(self.seed)
        self.faker: Faker = Faker()

        # Cache generated DataFrames
        self.customers_df: Optional[pd.DataFrame] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.orders_df: Optional[pd.DataFrame] = None

        logger.info(
            "Initialized SyntheticDataGenerator (customers=%d, products=%d, orders=%d, seed=%d)",
            self.num_customers,
            self.num_products,
            self.num_orders,
            self.seed,
        )

    def generate_customers(self, count: Optional[int] = None) -> pd.DataFrame:
        """Generate customer records with normally distributed age and Faker profiles.

        Schema:
            - customer_id: Unique string identifier (e.g., 'CUST_0000001')
            - name: Full name (Faker)
            - email: Email address (Faker)
            - age: Normally distributed around 35 (mean=35, std=10), clipped to [18, 80]
            - city: City name (Faker)
            - country: Country name (Faker)
            - registration_date: Date formatted string ('YYYY-MM-DD')

        Args:
            count: Number of customers to generate. Defaults to self.num_customers.

        Returns:
            A pandas DataFrame containing customer records.
        """
        n: int = count if count is not None else self.num_customers
        logger.info("Starting customer generation: %d records requested.", n)
        start_time: float = time.time()

        # 1. Customer IDs
        customer_ids: List[str] = [f"CUST_{i:07d}" for i in range(1, n + 1)]

        # 2. Age: Normal distribution around 35 (clipped between 18 and 80)
        raw_ages = np.random.normal(loc=35.0, scale=10.0, size=n)
        ages = np.clip(np.round(raw_ages), 18, 80).astype(int)

        # 3. Registration dates: Random days between 2021-01-01 and 2023-12-31
        start_reg = datetime(2021, 1, 1)
        end_reg = datetime(2023, 12, 31)
        total_days = (end_reg - start_reg).days
        random_day_offsets = np.random.randint(0, total_days + 1, size=n)
        base_timestamp = int(start_reg.timestamp())
        reg_dates = [
            (start_reg + timedelta(days=int(offset))).strftime("%Y-%m-%d")
            for offset in random_day_offsets
        ]

        # 4. Profile generation using Faker with batching for high throughput
        names: List[str] = []
        emails: List[str] = []
        cities: List[str] = []
        countries: List[str] = []

        # Create pools of cities and countries to accelerate 100k generation
        city_pool = [self.faker.city() for _ in range(500)]
        country_pool = [
            "United States",
            "Canada",
            "United Kingdom",
            "Germany",
            "France",
            "Australia",
            "Japan",
            "India",
            "Brazil",
            "Netherlands",
        ]
        sampled_cities = np.random.choice(city_pool, size=n)
        sampled_countries = np.random.choice(country_pool, size=n)

        # Generate unique names & emails in batches with progress bar
        batch_size = 10_000
        logger.info("Synthesizing names and emails in batches of %d...", batch_size)

        with tqdm(total=n, desc="Generating Customers", unit="records") as pbar:
            for i in range(0, n, batch_size):
                current_batch_size = min(batch_size, n - i)
                batch_first = [self.faker.first_name() for _ in range(current_batch_size)]
                batch_last = [self.faker.last_name() for _ in range(current_batch_size)]

                for idx in range(current_batch_size):
                    fn = batch_first[idx]
                    ln = batch_last[idx]
                    full_name = f"{fn} {ln}"
                    clean_email = f"{fn.lower()}.{ln.lower()}{i + idx + 1}@example.com"
                    names.append(full_name)
                    emails.append(clean_email)

                pbar.update(current_batch_size)

        self.customers_df = pd.DataFrame(
            {
                "customer_id": customer_ids,
                "name": names,
                "email": emails,
                "age": ages,
                "city": sampled_cities,
                "country": sampled_countries,
                "registration_date": reg_dates,
            }
        )

        elapsed = time.time() - start_time
        logger.info(
            "Customer generation completed: %d rows generated in %.2f seconds.",
            len(self.customers_df),
            elapsed,
        )
        return self.customers_df

    def generate_products(self, count: Optional[int] = None) -> pd.DataFrame:
        """Generate product catalog across 5 specified categories.

        Schema:
            - product_id: Unique string identifier (e.g., 'PROD_00001')
            - name: Descriptive product title
            - category: One of ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']
            - price: Float between 10.0 and 500.0 rounded to 2 decimals
            - stock: Random inventory count (0 to 1000)
            - rating: Float between 1.0 and 5.0 rounded to 1 decimal

        Args:
            count: Number of products to generate. Defaults to self.num_products.

        Returns:
            A pandas DataFrame containing product catalog records.
        """
        n: int = count if count is not None else self.num_products
        logger.info("Starting product generation: %d records requested.", n)
        start_time: float = time.time()

        categories: List[str] = ["Electronics", "Clothing", "Home", "Sports", "Books"]
        category_sample: np.ndarray = np.random.choice(categories, size=n)

        # Realistic adjectives and category-specific nouns
        adjectives = [
            "Premium",
            "Wireless",
            "Eco",
            "Smart",
            "Compact",
            "Ultra",
            "Classic",
            "Pro",
            "Deluxe",
            "Ergonomic",
            "Heavy-Duty",
            "Portable",
            "Modern",
        ]

        item_nouns: Dict[str, List[str]] = {
            "Electronics": ["Headphones", "Monitor", "Keyboard", "Speaker", "Charger", "Webcam", "Hub"],
            "Clothing": ["T-Shirt", "Jacket", "Jeans", "Hoodie", "Sneakers", "Cap", "Shorts"],
            "Home": ["Blender", "Cookware", "Lamp", "Air Purifier", "Rug", "Coffee Maker", "Vase"],
            "Sports": ["Yoga Mat", "Dumbbell Set", "Water Bottle", "Tent", "Backpack", "Rope"],
            "Books": ["Novel", "Handbook", "Guide", "Anthology", "Biography", "Cookbook", "Atlas"],
        }

        product_ids: List[str] = [f"PROD_{i:05d}" for i in range(1, n + 1)]

        # Generate price: Float between 10.0 and 500.0
        prices = np.round(np.random.uniform(10.0, 500.0, size=n), 2)

        # Generate stock: Integer between 0 and 1000
        stocks = np.random.randint(0, 1001, size=n)

        # Generate rating: Float between 1.0 and 5.0 (centered around 4.0)
        raw_ratings = np.random.normal(loc=4.1, scale=0.7, size=n)
        ratings = np.round(np.clip(raw_ratings, 1.0, 5.0), 1)

        product_names: List[str] = []
        with tqdm(total=n, desc="Generating Products", unit="records") as pbar:
            for i in range(n):
                cat = category_sample[i]
                adj = random.choice(adjectives)
                noun = random.choice(item_nouns[cat])
                product_names.append(f"{adj} {noun} #{i + 1}")
                pbar.update(1)

        self.products_df = pd.DataFrame(
            {
                "product_id": product_ids,
                "name": product_names,
                "category": category_sample,
                "price": prices,
                "stock": stocks,
                "rating": ratings,
            }
        )

        elapsed = time.time() - start_time
        logger.info(
            "Product generation completed: %d rows generated in %.2f seconds.",
            len(self.products_df),
            elapsed,
        )
        return self.products_df

    def generate_orders(
        self,
        count: Optional[int] = None,
        customers_df: Optional[pd.DataFrame] = None,
        products_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate orders with a Pareto distribution where 20% of customers make 80% of orders.

        Schema:
            - order_id: Unique string identifier (e.g., 'ORD_00000001')
            - customer_id: Foreign key to customer (Pareto weighted choice)
            - product_id: Foreign key to product
            - quantity: Integer quantity from 1 to 10
            - order_date: Date formatted string ('YYYY-MM-DD')

        Args:
            count: Number of orders to generate. Defaults to self.num_orders.
            customers_df: Customer DataFrame. If None, uses existing or generates one.
            products_df: Product DataFrame. If None, uses existing or generates one.

        Returns:
            A pandas DataFrame containing order records.
        """
        n: int = count if count is not None else self.num_orders
        logger.info("Starting order generation: %d records requested.", n)
        start_time: float = time.time()

        # Ensure customer data is present
        if customers_df is not None:
            self.customers_df = customers_df
        elif self.customers_df is None:
            self.generate_customers()

        assert self.customers_df is not None
        customer_ids: np.ndarray = self.customers_df["customer_id"].to_numpy()
        num_cust: int = len(customer_ids)

        # Ensure product data is present
        if products_df is not None:
            self.products_df = products_df
        elif self.products_df is None:
            self.generate_products()

        assert self.products_df is not None
        product_ids: np.ndarray = self.products_df["product_id"].to_numpy()

        # ======================================================================
        # Pareto Distribution: 80/20 rule (alpha = log(5) / log(4) ≈ 1.161)
        # 20% of customers will account for ~80% of generated orders
        # ======================================================================
        logger.info(
            "Calculating Pareto weights (alpha=1.161) across %d customers for 80/20 distribution...",
            num_cust,
        )
        pareto_alpha = 1.161
        pareto_weights = np.random.pareto(a=pareto_alpha, size=num_cust) + 1e-5
        customer_probabilities = pareto_weights / np.sum(pareto_weights)

        # Order date boundaries (e.g., across 2024)
        start_order_date = datetime(2024, 1, 1)
        end_order_date = datetime(2024, 12, 31)
        total_order_days = (end_order_date - start_order_date).days

        # Chunked generation for high performance & responsive tqdm progress bar
        batch_size = 200_000
        num_batches = (n + batch_size - 1) // batch_size
        logger.info("Synthesizing %d orders in %d batches...", n, num_batches)

        order_dfs: List[pd.DataFrame] = []

        with tqdm(total=n, desc="Generating Orders", unit="orders") as pbar:
            for b in range(num_batches):
                current_size = min(batch_size, n - (b * batch_size))
                offset = b * batch_size

                # Vectorized batch generation
                batch_order_ids = [f"ORD_{offset + i + 1:08d}" for i in range(current_size)]

                # Sample customer_id using the Pareto probability distribution
                batch_cust_ids = np.random.choice(
                    customer_ids, size=current_size, p=customer_probabilities
                )

                # Sample product_id uniformly
                batch_prod_ids = np.random.choice(product_ids, size=current_size)

                # Quantity: 1 to 10
                batch_quantities = np.random.randint(1, 11, size=current_size)

                # Order dates
                day_offsets = np.random.randint(0, total_order_days + 1, size=current_size)
                batch_dates = [
                    (start_order_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
                    for d in day_offsets
                ]

                batch_df = pd.DataFrame(
                    {
                        "order_id": batch_order_ids,
                        "customer_id": batch_cust_ids,
                        "product_id": batch_prod_ids,
                        "quantity": batch_quantities,
                        "order_date": batch_dates,
                    }
                )
                order_dfs.append(batch_df)
                pbar.update(current_size)

        self.orders_df = pd.concat(order_dfs, ignore_index=True)

        elapsed = time.time() - start_time
        logger.info(
            "Order generation completed: %d rows generated in %.2f seconds.",
            len(self.orders_df),
            elapsed,
        )
        return self.orders_df

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all three datasets (customers, products, and orders).

        Returns:
            Dictionary containing 'customers', 'products', and 'orders' DataFrames.
        """
        logger.info("Executing full synthetic data generation pipeline...")
        customers = self.generate_customers()
        products = self.generate_products()
        orders = self.generate_orders(customers_df=customers, products_df=products)

        return {
            "customers": customers,
            "products": products,
            "orders": orders,
        }

    def save_to_disk(
        self,
        output_dir: Optional[Path] = None,
        save_format: str = "csv",
    ) -> Dict[str, Path]:
        """Save generated DataFrames to disk in CSV, Parquet, or both formats.

        Args:
            output_dir: Target directory path. Defaults to RAW_DATA_DIR.
            save_format: Format string, one of 'csv', 'parquet', or 'both'.

        Returns:
            Dictionary mapping table names to saved file paths.
        """
        target_dir = output_dir or RAW_DATA_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        datasets = {
            "customers": self.customers_df
            if self.customers_df is not None
            else self.generate_customers(),
            "products": self.products_df
            if self.products_df is not None
            else self.generate_products(),
            "orders": self.orders_df
            if self.orders_df is not None
            else self.generate_orders(),
        }

        saved_paths: Dict[str, Path] = {}

        for name, df in datasets.items():
            if save_format in ("csv", "both"):
                csv_path = target_dir / f"{name}.csv"
                logger.info("Saving %s to CSV at %s...", name, csv_path)
                df.to_csv(csv_path, index=False)
                saved_paths[f"{name}_csv"] = csv_path

            if save_format in ("parquet", "both"):
                parquet_path = target_dir / f"{name}.parquet"
                logger.info("Saving %s to Parquet at %s...", name, parquet_path)
                df.to_parquet(parquet_path, index=False)
                saved_paths[f"{name}_parquet"] = parquet_path

        logger.info("All datasets successfully saved to %s", target_dir)
        return saved_paths


# Backward compatibility alias
EcommerceDataGenerator = SyntheticDataGenerator


def main() -> None:
    """CLI entrypoint for executing synthetic data generation."""
    logger.info("Starting synthetic data generation execution...")
    generator = SyntheticDataGenerator(
        num_customers=100_000,
        num_products=10_000,
        num_orders=1_000_000,
    )
    generator.generate_all()
    generator.save_to_disk(save_format="csv")
    logger.info("Data generation process successfully finished.")


if __name__ == "__main__":
    main()
