"""
Entry point for MATCH RECOGNIZE synthetic data generator.

Example usage:
    python main.py
    python main.py --config config.json
"""

import argparse
import json
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path to allow imports from match_recognize_datagen
sys.path.insert(0, str(Path(__file__).parent.parent))

from match_recognize_datagen import (
    GeneratorConfig,
    AttributeConfig,
    DefineSpec,
    IndependentCondition,
    ComparisonOperator,
    AttributeType,
    DistributionType,
    DataGenerator,
)


def create_example_config() -> GeneratorConfig:
    """Create an example configuration for demonstration."""
    return GeneratorConfig(
        initial_table_size=20,
        total_rows=50,
        num_columns=5,
        batch_sizes=[10, 10, 10],
        rows_per_window=5,
        time_window_size=30,
        attributes=[
            AttributeConfig(
                name="value",
                attr_type=AttributeType.NUMERICAL,
                min_value=1,
                max_value=100,
                distribution=DistributionType.NORMAL,
            ),
            AttributeConfig(
                name="category",
                attr_type=AttributeType.CATEGORICAL,
                categories=["A", "B", "C", "D"],
            ),
        ],
        output_dir="./output",
        output_format="csv",
    )


def load_config_from_file(filepath: str) -> GeneratorConfig:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to configuration JSON file
    
    Returns:
        GeneratorConfig instance
    
    Note:
        Full JSON deserialization with all config types is a future enhancement.
        For now, this is a placeholder.
    """
    with open(filepath, "r") as f:
        config_dict = json.load(f)

    # TODO: Fully deserialize from JSON
    # This would require custom JSON encoder/decoder for all config classes

    config = GeneratorConfig(
        initial_table_size=config_dict.get("initial_table_size", 1000),
        total_rows=config_dict.get("total_rows", 5000),
        num_columns=config_dict.get("num_columns", 5),
        batch_sizes=config_dict.get("batch_sizes", [1000, 1000, 1000]),
        rows_per_window=config_dict.get("rows_per_window", 100),
    )
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MATCH RECOGNIZE Synthetic Data Generator"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for generated files",
    )

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = create_example_config()

    # Override output directory if specified
    if args.output_dir != "./output":
        config.output_dir = args.output_dir

    print("=" * 70)
    print("MATCH RECOGNIZE Synthetic Data Generator")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Initial Table Size: {config.initial_table_size}")
    print(f"  Total Rows: {config.total_rows}")
    print(f"  Num Columns: {config.num_columns}")
    print(f"  Num Batches: {len(config.batch_sizes)}")
    print(f"  Batch Sizes: {config.batch_sizes}")
    print(f"  Rows per Window: {config.rows_per_window}")
    print(f"  Output Format: {config.output_format}")
    print(f"  Output Directory: {config.output_dir}")
    print()

    # Create and run generator
    generator = DataGenerator(config, seed=args.seed)
    full_table, batches = generator.generate()

    print(f"\nGeneration Summary:")
    print(f"  Full table: {len(full_table)} rows")
    print(f"  Batches: {len(batches)}")
    for i, batch in enumerate(batches, 1):
        print(f"    Batch {i}: {len(batch)} rows")


if __name__ == "__main__":
    main()
