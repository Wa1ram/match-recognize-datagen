"""
Example usage of MATCH RECOGNIZE synthetic data generator.

This demonstrates how to:
1. Configure table and batching parameters
2. Define a DEFINE clause with conditions
3. Generate synthetic data satisfying the constraints
4. Output generated batches
"""

from pathlib import Path
import sys

# Allow running this file directly without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from match_recognize_datagen import (
    GeneratorConfig,
    AttributeConfig,
    AttributeType,
    DistributionType,
    DefineSpec,
    IndependentCondition,
    DependentConditionPair,
    ComparisonOperator,
    DataGenerator,
)


def example_basic_generation():
    """Example 1: Basic generation without pattern or constraints."""
    print("Example 1: Basic Generation")
    print("-" * 50)

    config = GeneratorConfig(
        initial_table_size=1_000_000,
        total_rows=4_000_000,
        num_columns=4,
        batch_sizes=[1_000_000, 1_000_000, 1_000_000],
        rows_per_window=50,
        pattern_window_size=300.0,  # 50 rows spread over 300 seconds
        attributes=[
            AttributeConfig(
                name="price",
                attr_type=AttributeType.NUMERICAL,
                min_value=10,
                max_value=1000,
                distribution=DistributionType.UNIFORM,
            ),
            AttributeConfig(
                name="item_type",
                attr_type=AttributeType.CATEGORICAL,
                categories=["Electronics", "Clothing", "Food"],
            ),
        ],
        output_dir="./ex_output/example1",
    )

    generator = DataGenerator(config, seed=42)
    full_table, batches = generator.generate()
    print()


def example_with_independent_conditions():
    """Example 2: Generation with DEFINE independent conditions."""
    print("Example 2: DEFINE Independent Conditions")
    print("-" * 50)

    define_spec = DefineSpec(
        independent_conditions=[
            IndependentCondition(
                variable_name="V1",
                attribute_name="signal",
                operator=ComparisonOperator.GTE,
                value=60.0,
                selectivity=0.4,
            )
        ]
    )

    config = GeneratorConfig(
        initial_table_size=200,
        total_rows=1000,
        num_columns=5,
        batch_sizes=[200, 200, 200],
        rows_per_window=100,
        pattern_window_size=600.0,  # 100 rows spread over 600 seconds
        define_spec=define_spec,
        attributes=[
            AttributeConfig(
                name="signal",
                attr_type=AttributeType.NUMERICAL,
                min_value=0,
                max_value=100,
                distribution=DistributionType.NORMAL,
            ),
        ],
        output_dir="./output/example2",
    )

    generator = DataGenerator(config, seed=42)
    full_table, batches = generator.generate()
    print()


def example_with_conditions():
    """Example 3: Generation with DEFINE conditions and selectivity."""
    print("Example 3: With DEFINE Conditions")
    print("-" * 50)

    define_spec = DefineSpec(
        independent_conditions=[
            IndependentCondition(
                variable_name="V1",
                attribute_name="value",
                operator=ComparisonOperator.GT,
                value=50.0,
                selectivity=0.7,  # 70% of rows satisfy this
            ),
        ],
        pairwise_conditions=[
            DependentConditionPair(
                var1_name="V1",
                var2_name="V2",
                var1_attr="value",
                var2_attr="value",
                operator=ComparisonOperator.LT,
                threshold=10.0,
                selectivity=0.6,
            ),
        ],
    )

    config = GeneratorConfig(
        initial_table_size=150_000,
        total_rows=750_000,
        num_columns=5,
        batch_sizes=[200_000, 200_000, 200_000],
        rows_per_window=75,
        pattern_window_size=450.0,  # 75 rows spread over 450 seconds
        define_spec=define_spec,
        attributes=[
            AttributeConfig(
                name="value",
                attr_type=AttributeType.NUMERICAL,
                min_value=1,
                max_value=100,
                distribution=DistributionType.ZIPF,
            ),
            AttributeConfig(
                name="status",
                attr_type=AttributeType.CATEGORICAL,
                categories=["active", "inactive", "pending"],
            ),
        ],
        output_dir="./ex_output/example3",
    )

    generator = DataGenerator(config, seed=42)
    full_table, batches = generator.generate()
    print()


def example_full_specification():
    """Example 4: DEFINE specification with independent and pairwise conditions."""
    print("Example 4: Full DEFINE Specification")
    print("-" * 50)

    define_spec = DefineSpec(
        independent_conditions=[
            IndependentCondition(
                variable_name="Start",
                attribute_name="amount",
                operator=ComparisonOperator.GT,
                value=100.0,
                selectivity=0.5,
            ),
            IndependentCondition(
                variable_name="End",
                attribute_name="amount",
                operator=ComparisonOperator.LT,
                value=50.0,
                selectivity=0.5,
            ),
        ],
        pairwise_conditions=[
            DependentConditionPair(
                var1_name="Start",
                var2_name="End",
                var1_attr="amount",
                var2_attr="amount",
                operator=ComparisonOperator.GT,
                threshold=20.0,
                selectivity=0.6,
            ),
        ],
    )

    config = GeneratorConfig(
        initial_table_size=250,
        total_rows=1250,
        num_columns=6,
        batch_sizes=[250, 250, 250],
        rows_per_window=125,
        pattern_window_size=1200.0,  # 125 rows spread over 1200 seconds
        define_spec=define_spec,
        attributes=[
            AttributeConfig(
                name="amount",
                attr_type=AttributeType.NUMERICAL,
                min_value=0,
                max_value=200,
                distribution=DistributionType.NORMAL,
            ),
            AttributeConfig(
                name="event_type",
                attr_type=AttributeType.CATEGORICAL,
                categories=["Purchase", "Return", "Exchange"],
            ),
        ],
        output_dir="./output/example4",
    )

    generator = DataGenerator(config, seed=42)
    full_table, batches = generator.generate()
    print()


if __name__ == "__main__":
    #example_basic_generation()
    # example_with_independent_conditions()
    example_with_conditions()
    # example_full_specification()
    print("All examples completed!")
