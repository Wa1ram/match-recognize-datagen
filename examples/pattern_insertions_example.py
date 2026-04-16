"""Example for the insertion-first generator with config_a/config_b-style rules."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pattern_insertions_approach import (
    BatchPlan,
    ColumnConfig,
    ColumnType,
    ComparisonOperator,
    DefineConfig,
    DependentDistanceCondition,
    GapConfig,
    InsertionPattern,
    KleeneConfig,
    OutputConfig,
    Rule,
    RuleType,
    RunConfig,
    SpaceConfig,
    VariableConfig,
)
from pattern_insertions_approach.generator import PatternInsertionGenerator


def main() -> None:
    config_a = [
        Rule(type=RuleType.EXACT, value=40, probability=0.50),
        Rule(type=RuleType.RANGE, min=0, max=30, probability=0.10),
        Rule(type=RuleType.RANGE, min=31, max=100, probability=0.40),
    ]

    config_b = [
        Rule(type=RuleType.RANGE, min=10, max=50, probability=1.0),
    ]

    config_categorical = [
        Rule(type=RuleType.EXACT, value="A", probability=.6),
        Rule(type=RuleType.EXACT, value="C", probability=.4),
    ]

    columns = [
        ColumnConfig(name="value", column_type=ColumnType.NUMERICAL, min_value=0, max_value=100),
        ColumnConfig(name="kind", column_type=ColumnType.CATEGORICAL, categories=["A", "B", "C"]),
    ]

    variables = {
        "V1": VariableConfig(
            name="V1",
            column_distributions={
                "value": config_a,
                "kind": config_categorical,
            },
        ),
        "V2": VariableConfig(
            name="V2",
            kleene=KleeneConfig(min_length=1, max_length=3),
            column_distributions={"value": config_b},
        ),
    }

    pattern = InsertionPattern(
        variable_order=["V1", "V2"],
        gaps_between_variables=[GapConfig(min_length=1, max_length=3)],
    )

    define = DefineConfig(
        dependent_conditions=[
            DependentDistanceCondition(
                var_a="V1",
                var_b="V2",
                column_name="value",
                operator=ComparisonOperator.GTE,
                threshold=15,
                max_retries=100,
            )
        ]
    )

    run_config = RunConfig(
        columns=columns,
        variables=variables,
        pattern=pattern,
        define=define,
        batch_plan=BatchPlan(insertions_per_batch=[5, 5]),
        rows_per_window=10,
        time_window_size=120.0,
        space=SpaceConfig(before_first_variable=1, after_last_variable=2),
        output=OutputConfig(output_dir="./ex_output/pattern_insertions", output_format="csv"),
        random_seed=42,
    )

    generator = PatternInsertionGenerator(run_config)
    full_df, batches, report = generator.generate()
    generator.write_outputs(batches)

    print("Generated rows:", len(full_df))
    print("Dependent shortfalls:", len(report.dependent_shortfalls))
    print(full_df.head(10))


if __name__ == "__main__":
    main()
