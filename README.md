# E-Commerce Data Pipeline with PySpark

An end-to-end Python and PySpark data pipeline designed to generate synthetic relational e-commerce datasets (customers, products, orders, and order items) and run business intelligence analytics to uncover actionable insights.

---

## 📁 Project Structure

```text
genai-pyspark-pipeline1/
├── .gitignore                      # Git ignore rules for Python, Spark, and data files
├── README.md                       # Project overview and run guide
├── requirements.txt                # Python dependencies
├── data/
│   ├── raw/                        # Generated mock data (customers, products, orders, items)
│   │   └── .gitkeep
│   └── processed/                  # PySpark business analysis results
│       └── .gitkeep
├── notebooks/                      # Jupyter notebooks for interactive exploration
│   └── exploratory_analysis.ipynb
├── src/                            # Source code modules
│   ├── __init__.py                 # Package marker
│   ├── _init_.py                   # Package marker (alias)
│   ├── config.py                   # Centralized paths, schemas, and configurations
│   ├── data_generator.py           # Synthetic dataset generator using Faker & pandas
│   └── spark_analytics.py          # PySpark batch analytics processing engine
└── tests/                          # Unit and integration test suite
    ├── __init__.py
    └── test_pipeline.py            # Integrity and relational constraint tests
```

---

## 🚀 Key Features

1. **Synthetic Data Generation (`src/data_generator.py`)**:
   - Generates realistic, relationally consistent e-commerce data:
     - `customers.csv` (1,000 customers with segments: *Standard*, *Premium*, *VIP*)
     - `products.csv` (150 products across 6 retail categories with realistic pricing and profit margins)
     - `orders.csv` (5,000 transactions with status, dates, and payment methods)
     - `order_items.csv` (Multi-item shopping carts with quantity, discounts, and item net amounts)
   - Outputs both CSV and Parquet formats.

2. **PySpark Business Analytics (`src/spark_analytics.py`)**:
   - **SalesAnalytics utility**: Loads Parquet data and calculates top customers by revenue, category sales, and monthly revenue growth.
   - **Top Selling Products**: Identifies highest revenue generators and unit volumes.
   - **Category Performance**: Calculates net sales, total profit, and profit margin % per category.
   - **Customer Lifetime Value (CLV)**: Analyzes repeat order count, total customer spend, and segments users into *High Value*, *Medium Value*, and *Low Value* tiers.
   - **Monthly Sales Trend**: Evaluates Month-over-Month (MoM) revenue growth, return rates, and cancellation rates.
   - **Fulfillment & Payment Breakdown**: Summarizes order volumes across payment methods and delivery statuses.

3. **Software Engineering Standards**:
   - Strict **Type Annotations** (`typing`, `dataclasses`).
   - Detailed **Docstrings** (Google style) explaining parameters, returns, and logic.
   - Structured **Logging** with timestamps, levels, and dual file/console handlers.

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+ installed
- Java (JDK 8, 11, or 17) installed and added to `JAVA_HOME` (required for PySpark local execution)

### 1. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Pipeline

### Step 1: Generate Synthetic Data (Parquet)
Run the main orchestration script to generate mock datasets with full benchmarking:

```bash
python main.py
```

*Output files generated in `data/raw/`:*
- `customers.parquet` (100,000 records)
- `products.parquet` (10,000 records)
- `orders.parquet` (1,000,000 records)
- `customers.csv` / `customers.parquet`
- `products.csv` / `products.parquet`
- `orders.csv` / `orders.parquet`
- `order_items.csv` / `order_items.parquet`

### Step 2: Run PySpark Analytics
Process the raw data and generate business intelligence summaries in `data/processed/`:

```bash
python -m src.spark_analytics
```

*Processed result files generated in `data/processed/`:*
- `top_selling_products.csv` & `.parquet`
- `category_performance.csv` & `.parquet`
- `customer_lifetime_value.csv` & `.parquet`
- `monthly_sales_trend.csv` & `.parquet`
- `order_status_and_payment.csv` & `.parquet`

---

## ⚡ Hardware & File Format Benchmarking

Run comprehensive write/read latency, peak memory (`tracemalloc`), and CPU energy consumption benchmarks across 500,000 rows in **CSV, XLSX, Parquet, ORC, and Feather**:

```bash
python benchmark_formats.py
```

Compare Pandas and PySpark Parquet loading and sales aggregation on the generated 1,000,000-order dataset:

```bash
python benchmark_pandas_pyspark.py
```

---

## 🧪 Running Tests

Validate data consistency, relational integrity, and configurations using `pytest`:

```bash
pytest -v tests/
```

---

## 📊 Business Insights Derived

| Metric / Table | Business Value |
| :--- | :--- |
| **Top Selling Products** | Identifies inventory replenishment priorities and high-margin bestsellers. |
| **Category Performance** | Analyzes department profitability to optimize marketing budget allocation. |
| **Customer Lifetime Value** | Segment-specific retention programs and personalized VIP loyalty perks. |
| **Monthly Sales Trend** | Identifies seasonal trends and monitors return/cancellation spikes. |
| **Payment Distribution** | Guides payment gateway fee negotiations and checkout optimization. |
