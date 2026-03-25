"""
Output writers for generated data in various formats.

Supports: Parquet (primary), CSV, SQL INSERT statements
"""

import os
from pathlib import Path
from typing import List
import pandas as pd
from .config import GeneratorConfig


class OutputWriter:
    """Handles output of generated data to files."""

    def __init__(self, config: GeneratorConfig):
        """
        Initialize output writer.
        
        Args:
            config: GeneratorConfig instance
        """
        self.config = config
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def write_batches(self, batches: List[pd.DataFrame]):
        """
        Write batches to output files.
        
        Args:
            batches: List of DataFrames, one per batch
        """
        for i, batch_df in enumerate(batches):
            batch_num = i + 1
            self._write_batch(batch_df, batch_num)

    def write_initial_table(self, initial_table_df: pd.DataFrame):
        """
        Write initial table to output format.

        Args:
            initial_table_df: DataFrame containing initial table rows
        """
        filename_base = "initial_table"

        if self.config.output_format == "parquet":
            self._write_parquet(initial_table_df, filename_base)
        elif self.config.output_format == "csv":
            self._write_csv(initial_table_df, filename_base)
        elif self.config.output_format == "sql":
            self._write_sql(initial_table_df, filename_base)
        else:
            raise ValueError(f"Unknown output format: {self.config.output_format}")

    def _write_batch(self, batch_df: pd.DataFrame, batch_num: int):
        """
        Write a single batch to output format.
        
        Args:
            batch_df: DataFrame for this batch
            batch_num: Batch number (1-indexed)
        """
        filename_base = f"batch_{batch_num:03d}"

        if self.config.output_format == "parquet":
            self._write_parquet(batch_df, filename_base)
        elif self.config.output_format == "csv":
            self._write_csv(batch_df, filename_base)
        elif self.config.output_format == "sql":
            self._write_sql(batch_df, filename_base)
        else:
            raise ValueError(f"Unknown output format: {self.config.output_format}")

    def _write_parquet(self, batch_df: pd.DataFrame, filename_base: str):
        """
        Write batch to Parquet format.
        
        Args:
            batch_df: DataFrame to write
            filename_base: Base filename (without extension)
        """
        filepath = os.path.join(self.config.output_dir, f"{filename_base}.parquet")
        batch_df.to_parquet(filepath, index=False)
        print(f"  Written: {filepath}")

    def _write_csv(self, batch_df: pd.DataFrame, filename_base: str):
        """
        Write batch to CSV format.
        
        Args:
            batch_df: DataFrame to write
            filename_base: Base filename (without extension)
        """
        filepath = os.path.join(self.config.output_dir, f"{filename_base}.csv")
        batch_df.to_csv(filepath, index=False)
        print(f"  Written: {filepath}")

    def _write_sql(self, batch_df: pd.DataFrame, filename_base: str):
        """
        Write batch as SQL INSERT statements.
        
        Args:
            batch_df: DataFrame to write
            filename_base: Base filename (without extension)
        """
        filepath = os.path.join(self.config.output_dir, f"{filename_base}.sql")

        # Infer table name from batch number or use default
        table_name = "data_table"  # TODO: Make configurable

        with open(filepath, "w") as f:
            # Write header comment
            f.write(f"-- Batch {filename_base}\n")
            f.write(f"-- {len(batch_df)} rows\n\n")

            # Write INSERT statements
            for _, row in batch_df.iterrows():
                columns = list(batch_df.columns)
                values = []

                for col in columns:
                    val = row[col]
                    if isinstance(val, str):
                        values.append(f"'{val}'")
                    elif pd.isna(val):
                        values.append("NULL")
                    else:
                        values.append(str(val))

                insert_stmt = (
                    f"INSERT INTO {table_name} "
                    f"({', '.join(columns)}) "
                    f"VALUES ({', '.join(values)});"
                )
                f.write(insert_stmt + "\n")

        print(f"  Written: {filepath}")

    @staticmethod
    def write_full_table(
        df: pd.DataFrame,
        output_dir: str,
        filename: str = "full_table",
        format: str = "parquet",
    ):
        """
        Write full table to file (convenience method).
        
        Args:
            df: DataFrame to write
            output_dir: Output directory
            filename: Filename without extension
            format: Output format (parquet, csv, sql)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            filepath = os.path.join(output_dir, f"{filename}.parquet")
            df.to_parquet(filepath, index=False)
        elif format == "csv":
            filepath = os.path.join(output_dir, f"{filename}.csv")
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Written full table to: {filepath}")
