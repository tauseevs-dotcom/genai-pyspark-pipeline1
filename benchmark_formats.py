"""File Format Performance and Hardware Metrics Benchmark.

This script benchmarks tabular storage formats (CSV, XLSX, Parquet, ORC, Feather)
on a 500,000 row dataset, measuring:
1. File size on disk (MB)
2. Serialization (Write) latency (seconds)
3. Deserialization (Read) latency (seconds)
4. Peak RAM usage via `tracemalloc` (MB)
5. CPU Execution Time via `time.process_time` (seconds)
6. Estimated CPU energy consumption (Watt-hours) based on 65W TDP:
   Wh = (CPU_time * 65W) / 3600
7. Percentage efficiency gains / savings relative to standard CSV baseline.
"""

from dataclasses import dataclass
import gc
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("format_benchmark")

# Constants
DEFAULT_ROW_COUNT: int = 500_000
CPU_TDP_WATTS: float = 65.0  # Assumed standard desktop/laptop processor TDP
BENCHMARK_DIR: Path = Path(__file__).resolve().parent / "data" / "benchmark"


@dataclass
class BenchmarkResult:
    """Holds all performance and hardware metrics for a single format."""

    format_name: str
    extension: str
    file_size_mb: float
    write_wall_time: float
    write_cpu_time: float
    write_peak_mem_mb: float
    read_wall_time: float
    read_cpu_time: float
    read_peak_mem_mb: float

    @property
    def total_wall_time(self) -> float:
        """Total wall-clock time for write + read in seconds."""
        return self.write_wall_time + self.read_wall_time

    @property
    def total_cpu_time(self) -> float:
        """Total CPU process time for write + read in seconds."""
        return self.write_cpu_time + self.read_cpu_time

    @property
    def peak_memory_mb(self) -> float:
        """Highest peak memory recorded between write and read cycles."""
        return max(self.write_peak_mem_mb, self.read_peak_mem_mb)

    @property
    def energy_consumption_wh(self) -> float:
        """Estimated energy consumed in Watt-hours (Wh): (CPU_time * TDP) / 3600."""
        return (self.total_cpu_time * CPU_TDP_WATTS) / 3600.0


def generate_benchmark_dataframe(num_rows: int = DEFAULT_ROW_COUNT, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic test DataFrame with 500,000 records.

    Columns:
        - id: Integer sequential identifier
        - name: Synthetic person name
        - email: Synthetic email address
        - amount: Float transaction monetary amount (5.00 - 999.00)
        - date: String formatted date ('YYYY-MM-DD')
        - category: Categorical product sector

    Args:
        num_rows: Number of records to generate. Default 500,000.
        seed: Random seed for repeatability.

    Returns:
        Populated pandas DataFrame.
    """
    logger.info("Generating synthetic benchmark DataFrame with %d rows...", num_rows)
    t0 = time.perf_counter()

    np.random.seed(seed)

    first_names = np.array([
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph",
        "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel", "Lisa", "Matthew",
    ])
    last_names = np.array([
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    ])
    categories = np.array(["Electronics", "Clothing", "Home", "Sports", "Books"])

    # Fast vectorized synthesis
    ids = np.arange(1, num_rows + 1, dtype=np.int32)
    fn_sample = np.random.choice(first_names, size=num_rows)
    ln_sample = np.random.choice(last_names, size=num_rows)

    names = [f"{f} {l}" for f, l in zip(fn_sample, ln_sample)]
    emails = [f"{f.lower()}.{l.lower()}{i}@example.com" for i, (f, l) in enumerate(zip(fn_sample, ln_sample), 1)]
    amounts = np.round(np.random.uniform(5.0, 999.0, size=num_rows), 2)
    cats = np.random.choice(categories, size=num_rows)

    base_date = pd.Timestamp("2024-01-01")
    random_days = np.random.randint(0, 365, size=num_rows)
    dates = (base_date + pd.to_timedelta(random_days, unit="D")).strftime("%Y-%m-%d")

    df = pd.DataFrame(
        {
            "id": ids,
            "name": names,
            "email": emails,
            "amount": amounts,
            "date": dates,
            "category": cats,
        }
    )

    elapsed = time.perf_counter() - t0
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    logger.info("DataFrame created in %.2fs. In-memory footprint: %.2f MB", elapsed, mem_mb)
    return df


def measure_operation(operation_fn: Callable[[], Any]) -> Tuple[Any, float, float, float]:
    """Execute an operation while tracking wall time, CPU time, and peak memory.

    Args:
        operation_fn: Callable executing the write or read function.

    Returns:
        Tuple of (result, wall_time_seconds, cpu_time_seconds, peak_memory_mb).
    """
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()

    start_cpu = time.process_time()
    start_wall = time.perf_counter()

    result = operation_fn()

    wall_time = time.perf_counter() - start_wall
    cpu_time = time.process_time() - start_cpu

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024.0**2)

    return result, wall_time, cpu_time, peak_mb


def benchmark_single_format(
    format_name: str,
    extension: str,
    write_fn: Callable[[pd.DataFrame, Path], None],
    read_fn: Callable[[Path], pd.DataFrame],
    df: pd.DataFrame,
    target_dir: Path,
) -> BenchmarkResult:
    """Run write and read benchmarks for a specific tabular storage format.

    Args:
        format_name: Descriptive name of the file format (e.g. 'Parquet (PyArrow)').
        extension: File extension (e.g. '.parquet').
        write_fn: Function to serialize DataFrame to disk.
        read_fn: Function to deserialize file into DataFrame.
        df: Input DataFrame to serialize.
        target_dir: Directory to save temporary test files.

    Returns:
        BenchmarkResult populated with performance metrics.
    """
    filepath = target_dir / f"benchmark_data{extension}"
    if filepath.exists():
        filepath.unlink()

    logger.info("--> Testing %s ...", format_name)

    # 1. Measure Write
    _, write_wall, write_cpu, write_peak_mem = measure_operation(lambda: write_fn(df, filepath))
    file_size_mb = filepath.stat().st_size / (1024.0**2)

    logger.info(
        "    [Write] Wall: %.3fs | CPU: %.3fs | Peak RAM: %.2f MB | Size: %.2f MB",
        write_wall,
        write_cpu,
        write_peak_mem,
        file_size_mb,
    )

    # 2. Measure Read
    read_df, read_wall, read_cpu, read_peak_mem = measure_operation(lambda: read_fn(filepath))

    logger.info(
        "    [Read]  Wall: %.3fs | CPU: %.3fs | Peak RAM: %.2f MB | Rows: %d",
        read_wall,
        read_cpu,
        read_peak_mem,
        len(read_df),
    )

    # Clean up file after verification
    try:
        filepath.unlink()
    except Exception:
        pass

    return BenchmarkResult(
        format_name=format_name,
        extension=extension,
        file_size_mb=file_size_mb,
        write_wall_time=write_wall,
        write_cpu_time=write_cpu,
        write_peak_mem_mb=write_peak_mem,
        read_wall_time=read_wall,
        read_cpu_time=read_cpu,
        read_peak_mem_mb=read_peak_mem,
    )


def run_all_benchmarks(
    num_rows: int = DEFAULT_ROW_COUNT,
    include_xlsx: bool = True,
) -> List[BenchmarkResult]:
    """Orchestrate benchmarks across CSV, XLSX, Parquet, ORC, and Feather.

    Args:
        num_rows: Number of records to test. Default: 500,000.
        include_xlsx: Whether to test Excel format (note: Excel is slow on 500k rows).

    Returns:
        List of BenchmarkResult objects.
    """
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_benchmark_dataframe(num_rows=num_rows)

    results: List[BenchmarkResult] = []

    # 1. CSV
    results.append(
        benchmark_single_format(
            format_name="CSV",
            extension=".csv",
            write_fn=lambda d, p: d.to_csv(p, index=False),
            read_fn=lambda p: pd.read_csv(p),
            df=df,
            target_dir=BENCHMARK_DIR,
        )
    )

    # 2. Parquet (PyArrow Engine)
    results.append(
        benchmark_single_format(
            format_name="Parquet (PyArrow)",
            extension=".parquet",
            write_fn=lambda d, p: d.to_parquet(p, index=False, engine="pyarrow", compression="snappy"),
            read_fn=lambda p: pd.read_parquet(p, engine="pyarrow"),
            df=df,
            target_dir=BENCHMARK_DIR,
        )
    )

    # 3. Parquet (fastparquet Engine)
    try:
        import fastparquet
        results.append(
            benchmark_single_format(
                format_name="Parquet (fastparquet)",
                extension="_fastparquet.parquet",
                write_fn=lambda d, p: d.to_parquet(p, index=False, engine="fastparquet", compression="snappy"),
                read_fn=lambda p: pd.read_parquet(p, engine="fastparquet"),
                df=df,
                target_dir=BENCHMARK_DIR,
            )
        )
    except ImportError:
        logger.warning("fastparquet is not installed; skipping fastparquet variant.")

    # 4. ORC (Optimized Row Columnar via PyArrow)
    results.append(
        benchmark_single_format(
            format_name="ORC",
            extension=".orc",
            write_fn=lambda d, p: d.to_orc(p, index=False),
            read_fn=lambda p: pd.read_orc(p),
            df=df,
            target_dir=BENCHMARK_DIR,
        )
    )

    # 5. Feather (Arrow IPC)
    results.append(
        benchmark_single_format(
            format_name="Feather",
            extension=".feather",
            write_fn=lambda d, p: d.to_feather(p),
            read_fn=lambda p: pd.read_feather(p),
            df=df,
            target_dir=BENCHMARK_DIR,
        )
    )

    # 6. XLSX (Excel OpenPyXL)
    if include_xlsx:
        logger.info("--> Testing Excel XLSX (Note: 500,000 rows can require significant time & memory)...")
        try:
            results.append(
                benchmark_single_format(
                    format_name="XLSX (Excel)",
                    extension=".xlsx",
                    write_fn=lambda d, p: d.to_excel(p, index=False, engine="openpyxl"),
                    read_fn=lambda p: pd.read_excel(p, engine="openpyxl"),
                    df=df,
                    target_dir=BENCHMARK_DIR,
                )
            )
        except Exception as e:
            logger.error("Failed or timed out testing XLSX: %s", e)

    # Clean up benchmark workspace
    if BENCHMARK_DIR.exists():
        shutil.rmtree(BENCHMARK_DIR, ignore_errors=True)

    return results


def print_comparison_tables(results: List[BenchmarkResult]) -> None:
    """Print beautifully formatted comparison tables and percentage savings vs CSV.

    Args:
        results: List of completed BenchmarkResult instances.
    """
    if not results:
        logger.error("No benchmark results to display.")
        return

    # Find baseline (CSV)
    baseline = next((r for r in results if r.format_name.upper().startswith("CSV")), results[0])

    print("\n" + "=" * 120)
    print(f"        FILE FORMAT BENCHMARK ON 500,000 ROWS WITH HARDWARE METRICS (CPU TDP: {CPU_TDP_WATTS:.0f}W)")
    print("=" * 120)

    # Table 1: Raw Measurements
    headers_1 = [
        "Format",
        "Size (MB)",
        "Write (s)",
        "Read (s)",
        "Total (s)",
        "CPU Time (s)",
        "Peak RAM (MB)",
        "Energy (Wh)",
    ]
    fmt_1 = "{:<22} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12} | {:>13} | {:>12}"
    print(fmt_1.format(*headers_1))
    print("-" * 120)

    for r in results:
        print(
            fmt_1.format(
                r.format_name,
                f"{r.file_size_mb:.2f}",
                f"{r.write_wall_time:.3f}",
                f"{r.read_wall_time:.3f}",
                f"{r.total_wall_time:.3f}",
                f"{r.total_cpu_time:.3f}",
                f"{r.peak_memory_mb:.2f}",
                f"{r.energy_consumption_wh:.4f}",
            )
        )
    print("=" * 120)

    # Table 2: Percentage Savings vs CSV Baseline
    print("\n" + "=" * 120)
    print(f"               PERCENTAGE SAVINGS VS CSV BASELINE (Positive % = Better / Faster / Smaller)")
    print("=" * 120)

    headers_2 = [
        "Format",
        "Disk Savings %",
        "Write Speedup %",
        "Read Speedup %",
        "RAM Savings %",
        "Energy Savings %",
    ]
    fmt_2 = "{:<22} | {:>14} | {:>15} | {:>14} | {:>13} | {:>16}"
    print(fmt_2.format(*headers_2))
    print("-" * 120)

    def calc_saving(base_val: float, test_val: float) -> str:
        """Calculate percentage saving: positive means test_val is lower/better."""
        if base_val == 0:
            return "N/A"
        pct = ((base_val - test_val) / base_val) * 100.0
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"

    for r in results:
        disk_sav = calc_saving(baseline.file_size_mb, r.file_size_mb)
        write_sav = calc_saving(baseline.write_wall_time, r.write_wall_time)
        read_sav = calc_saving(baseline.read_wall_time, r.read_wall_time)
        ram_sav = calc_saving(baseline.peak_memory_mb, r.peak_memory_mb)
        energy_sav = calc_saving(baseline.energy_consumption_wh, r.energy_consumption_wh)

        print(
            fmt_2.format(
                r.format_name,
                disk_sav if r != baseline else "Baseline (0%)",
                write_sav if r != baseline else "Baseline (0%)",
                read_sav if r != baseline else "Baseline (0%)",
                ram_sav if r != baseline else "Baseline (0%)",
                energy_sav if r != baseline else "Baseline (0%)",
            )
        )

    print("=" * 120)
    print("\nFormula: Energy (Wh) = (Total_CPU_Time_sec * 65.0W) / 3600.0\n")


def main() -> None:
    """CLI entrypoint to execute format benchmark."""
    logger.info("Starting tabular file format hardware metrics benchmark...")
    try:
        results = run_all_benchmarks(num_rows=DEFAULT_ROW_COUNT, include_xlsx=True)
        print_comparison_tables(results)
        logger.info("Benchmark finished successfully.")
    except KeyboardInterrupt:
        logger.warning("\nBenchmark cancelled by user.")
        sys.exit(130)
    except Exception as exc:
        logger.critical("Benchmark encountered an unexpected error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

