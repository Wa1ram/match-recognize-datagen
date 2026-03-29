import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from match_recognize_datagen import (
    AttributeConfig,
    AttributeType,
    ComparisonOperator,
    DataGenerator,
    DependentConditionPair,
    DefineSpec,
    DistributionType,
    GeneratorConfig,
    IndependentCondition,
)


class TestParquetOutput(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = self.tmpdir.name

        self.config = GeneratorConfig(
            initial_table_size=20,
            total_rows=50,
            num_columns=4,
            batch_sizes=[10, 10, 10],
            rows_per_window=5,
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
                    categories=["A", "B", "C"],
                ),
            ],
            output_dir=self.output_dir,
            output_format="parquet",
        )

        self.generator = DataGenerator(self.config, seed=123)
        self.full_table, self.in_memory_batches = self.generator.generate()
        self.in_memory_initial_table = self.full_table.iloc[: self.config.initial_table_size].copy()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _batch_files(self):
        return sorted(Path(self.output_dir).glob("batch_*.parquet"))

    def _read_batches_from_parquet(self):
        return [pd.read_parquet(path) for path in self._batch_files()]

    def _initial_table_file(self):
        return Path(self.output_dir) / "initial_table.parquet"

    def _read_initial_table_from_parquet(self):
        return pd.read_parquet(self._initial_table_file())

    def test_creates_initial_table_parquet_file(self):
        self.assertTrue(self._initial_table_file().exists())

    def test_initial_table_row_count_is_correct(self):
        initial_table_df = self._read_initial_table_from_parquet()
        self.assertEqual(len(initial_table_df), self.config.initial_table_size)

    def test_initial_table_matches_in_memory_slice(self):
        initial_table_df = self._read_initial_table_from_parquet()
        assert_frame_equal(
            initial_table_df.reset_index(drop=True),
            self.in_memory_initial_table.reset_index(drop=True),
            check_dtype=False,
        )

    def test_creates_expected_number_of_parquet_files(self):
        files = self._batch_files()
        self.assertEqual(len(files), len(self.config.batch_sizes))
        self.assertEqual(
            [p.name for p in files],
            ["batch_001.parquet", "batch_002.parquet", "batch_003.parquet"],
        )

    def test_each_parquet_batch_has_expected_row_count(self):
        parquet_batches = self._read_batches_from_parquet()
        actual_counts = [len(df) for df in parquet_batches]
        self.assertEqual(actual_counts, self.config.batch_sizes)

    def test_parquet_schema_has_required_columns(self):
        parquet_batches = self._read_batches_from_parquet()
        expected_cols = {"id", "timestamp", "value", "category"}

        for df in parquet_batches:
            self.assertEqual(set(df.columns), expected_cols)

    def test_parquet_batches_match_in_memory_batches(self):
        parquet_batches = self._read_batches_from_parquet()
        self.assertEqual(len(parquet_batches), len(self.in_memory_batches))

        for parquet_df, memory_df in zip(parquet_batches, self.in_memory_batches):
            assert_frame_equal(
                parquet_df.reset_index(drop=True),
                memory_df.reset_index(drop=True),
                check_dtype=False,
            )

    def test_ids_are_continuous_across_all_written_batches(self):
        parquet_batches = self._read_batches_from_parquet()
        concatenated = pd.concat(parquet_batches, ignore_index=True)

        first_written_id = self.config.initial_table_size + 1
        last_written_id = self.config.total_rows
        expected_ids = list(range(first_written_id, last_written_id + 1))
        actual_ids = concatenated["id"].astype(int).tolist()

        self.assertEqual(actual_ids, expected_ids)

    def test_basic_value_constraints_in_parquet(self):
        parquet_batches = self._read_batches_from_parquet()
        concatenated = pd.concat(parquet_batches, ignore_index=True)

        self.assertFalse(concatenated["id"].isna().any())
        self.assertFalse(concatenated["timestamp"].isna().any())
        self.assertFalse(concatenated["value"].isna().any())
        self.assertFalse(concatenated["category"].isna().any())

        self.assertTrue((concatenated["timestamp"] >= 0).all())
        self.assertTrue((concatenated["value"] >= 1).all())
        self.assertTrue((concatenated["value"] <= 100).all())
        self.assertTrue(concatenated["category"].isin(["A", "B", "C"]).all())

    def test_categorical_independent_eq_condition_is_applied(self):
        cfg = GeneratorConfig(
            initial_table_size=20,
            total_rows=50,
            num_columns=4,
            batch_sizes=[10, 10, 10],
            rows_per_window=5,
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
                    categories=["A", "B", "C"],
                ),
            ],
            define_spec=DefineSpec(
                independent_conditions=[
                    IndependentCondition(
                        variable_name="V1",
                        attribute_name="category",
                        operator=ComparisonOperator.EQ,
                        value="B",
                        selectivity=1.0,
                    )
                ]
            ),
            output_dir=self.output_dir,
            output_format="parquet",
        )
        generator = DataGenerator(cfg, seed=123)
        constrained = generator.apply_define_constraints(generator.generate_full_table())

        self.assertTrue((constrained["category"] == "B").all())

    def test_categorical_independent_neq_condition_is_applied(self):
        cfg = GeneratorConfig(
            initial_table_size=20,
            total_rows=50,
            num_columns=4,
            batch_sizes=[10, 10, 10],
            rows_per_window=5,
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
                    categories=["A", "B", "C"],
                ),
            ],
            define_spec=DefineSpec(
                independent_conditions=[
                    IndependentCondition(
                        variable_name="V1",
                        attribute_name="category",
                        operator=ComparisonOperator.NEQ,
                        value="A",
                        selectivity=1.0,
                    )
                ]
            ),
            output_dir=self.output_dir,
            output_format="parquet",
        )
        generator = DataGenerator(cfg, seed=123)
        constrained = generator.apply_define_constraints(generator.generate_full_table())

        self.assertTrue((constrained["category"] != "A").all())

    def test_numerical_independent_eq_selectivity_half(self):
        cfg = GeneratorConfig(
            initial_table_size=200,
            total_rows=1000,
            num_columns=4,
            batch_sizes=[400, 400],
            rows_per_window=10,
            attributes=[
                AttributeConfig(
                    name="value",
                    attr_type=AttributeType.NUMERICAL,
                    min_value=1,
                    max_value=100,
                    distribution=DistributionType.UNIFORM,
                ),
                AttributeConfig(
                    name="category",
                    attr_type=AttributeType.CATEGORICAL,
                    categories=["A", "B", "C"],
                ),
            ],
            define_spec=DefineSpec(
                independent_conditions=[
                    IndependentCondition(
                        variable_name="V1",
                        attribute_name="value",
                        operator=ComparisonOperator.EQ,
                        value=42.0,
                        selectivity=0.5,
                    )
                ]
            ),
            output_dir=self.output_dir,
            output_format="parquet",
        )

        generator = DataGenerator(cfg, seed=123)
        constrained = generator.apply_define_constraints(generator.generate_full_table())

        ratio_eq_42 = (constrained["value"] == 42.0).mean()
        self.assertGreater(ratio_eq_42, 0.48)
        self.assertLess(ratio_eq_42, 0.52)

    def test_generation_directly_respects_multiple_independent_categorical_conditions(self):
        cfg = GeneratorConfig(
            initial_table_size=2000,
            total_rows=20000,
            num_columns=4,
            batch_sizes=[9000, 9000],
            rows_per_window=20,
            attributes=[
                AttributeConfig(
                    name="value",
                    attr_type=AttributeType.NUMERICAL,
                    min_value=0,
                    max_value=100,
                    distribution=DistributionType.UNIFORM,
                ),
                AttributeConfig(
                    name="category",
                    attr_type=AttributeType.CATEGORICAL,
                    categories=["A", "B", "C"],
                ),
            ],
            define_spec=DefineSpec(
                independent_conditions=[
                    IndependentCondition(
                        variable_name="V1",
                        attribute_name="category",
                        operator=ComparisonOperator.EQ,
                        value="B",
                        selectivity=0.60,
                    ),
                    IndependentCondition(
                        variable_name="V2",
                        attribute_name="category",
                        operator=ComparisonOperator.NEQ,
                        value="A",
                        selectivity=0.90,
                    ),
                ]
            ),
            output_dir=self.output_dir,
            output_format="parquet",
        )

        generator = DataGenerator(cfg, seed=123)
        generated = generator.generate_full_table()

        ratio_eq_b = (generated["category"] == "B").mean()
        ratio_neq_a = (generated["category"] != "A").mean()

        self.assertAlmostEqual(ratio_eq_b, 0.60, places=6)
        self.assertAlmostEqual(ratio_neq_a, 0.90, places=6)

    def test_pairwise_dependent_selectivity_is_applied(self):
        cfg = GeneratorConfig(
            initial_table_size=20,
            total_rows=200,
            num_columns=4,
            batch_sizes=[90, 90],
            rows_per_window=10,
            attributes=[
                AttributeConfig(
                    name="value",
                    attr_type=AttributeType.NUMERICAL,
                    min_value=0,
                    max_value=1000,
                    distribution=DistributionType.UNIFORM,
                ),
                AttributeConfig(
                    name="category",
                    attr_type=AttributeType.CATEGORICAL,
                    categories=["A", "B", "C"],
                ),
            ],
            define_spec=DefineSpec(
                pairwise_conditions=[
                    DependentConditionPair(
                        var1_name="V1",
                        var2_name="V2",
                        var1_attr="value",
                        var2_attr="value",
                        operator=ComparisonOperator.LTE,
                        threshold=5.0,
                        selectivity=0.25,
                    )
                ]
            ),
            output_dir=self.output_dir,
            output_format="parquet",
        )

        generator = DataGenerator(cfg, seed=123)
        constrained = generator.apply_define_constraints(generator.generate_full_table())

        values = constrained["value"].tolist()
        n = len(values)
        total_pairs = n * (n - 1) // 2
        satisfied_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(values[i] - values[j]) <= 5.0:
                    satisfied_pairs += 1

        expected_pairs = int(round(0.25 * total_pairs))
        self.assertEqual(satisfied_pairs, expected_pairs)

    def test_independent_selectivity_stays_stable_with_pairwise_overlap(self):
        cfg = GeneratorConfig(             
            initial_table_size=100,
            total_rows=500,
            num_columns=4,
            batch_sizes=[200, 200],
            rows_per_window=10,
            attributes=[
                AttributeConfig(
                    name="value",
                    attr_type=AttributeType.NUMERICAL,
                    min_value=0,
                    max_value=100,
                    distribution=DistributionType.UNIFORM,
                ),
                AttributeConfig(
                    name="category",
                    attr_type=AttributeType.CATEGORICAL,
                    categories=["A", "B", "C"],
                ),
            ],
            define_spec=DefineSpec(
                independent_conditions=[
                    IndependentCondition(
                        variable_name="V1",
                        attribute_name="value",
                        operator=ComparisonOperator.GTE,
                        value=50.0,
                        selectivity=0.4,
                    )
                ],
                pairwise_conditions=[
                    DependentConditionPair(
                        var1_name="V1",
                        var2_name="V2",
                        var1_attr="value",
                        var2_attr="value",
                        operator=ComparisonOperator.LTE,
                        threshold=5.0,
                        selectivity=0.3,
                    )
                ],
            ),
            output_dir=self.output_dir,
            output_format="parquet",
        )

        generator = DataGenerator(cfg, seed=123)
        constrained = generator.apply_define_constraints(generator.generate_full_table())

        ratio_ge_50 = (constrained["value"] >= 50.0).mean()
        self.assertAlmostEqual(ratio_ge_50, 0.4, places=6)

    def test_generation_directly_respects_multiple_independent_ranges(self):
        cfg = GeneratorConfig(
            initial_table_size=200,
            total_rows=2000,
            num_columns=4,
            batch_sizes=[900, 900],
            rows_per_window=20,
            attributes=[
                AttributeConfig(
                    name="value",
                    attr_type=AttributeType.NUMERICAL,
                    min_value=0,
                    max_value=100,
                    distribution=DistributionType.UNIFORM,
                ),
                AttributeConfig(
                    name="category",
                    attr_type=AttributeType.CATEGORICAL,
                    categories=["A", "B", "C"],
                ),
            ],
            define_spec=DefineSpec(
                independent_conditions=[
                    IndependentCondition(
                        variable_name="V1",
                        attribute_name="value",
                        operator=ComparisonOperator.EQ,
                        value=50.0,
                        selectivity=0.50,
                    ),
                    IndependentCondition(
                        variable_name="V2",
                        attribute_name="value",
                        operator=ComparisonOperator.GT,
                        value=20.0,
                        selectivity=0.95,
                    ),
                ]
            ),
            output_dir=self.output_dir,
            output_format="parquet",
        )

        generator = DataGenerator(cfg, seed=123)
        generated = generator.generate_full_table()

        ratio_eq_50 = (generated["value"] == 50.0).mean()
        ratio_gt_20 = (generated["value"] > 20.0).mean()

        self.assertGreater(ratio_eq_50, 0.46)
        self.assertLess(ratio_eq_50, 0.54)
        self.assertGreater(ratio_gt_20, 0.91)
        self.assertLess(ratio_gt_20, 0.98)


if __name__ == "__main__":
    unittest.main(verbosity=2)
