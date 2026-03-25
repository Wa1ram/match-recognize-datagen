"""
Core data generation logic for MATCH RECOGNIZE synthetic data.
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from .config import (
    GeneratorConfig,
    AttributeType,
    DistributionType,
    PatternSpec,
)
from .pattern import PatternMatcher
from .define import ConditionEvaluator, SelectivityApplier


class DataGenerator:
    """
    Main synthetic data generator for MATCH RECOGNIZE queries.
    
    Workflow:
    1. Generate full table (initial + all batches)
    2. Apply pattern constraints
    3. Apply DEFINE conditions
    4. Slice into separate batch files
    5. Output to target format
    """

    def __init__(self, config: GeneratorConfig, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            config: GeneratorConfig instance
            seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = random.Random(seed)
        self.pattern_matcher = (
            PatternMatcher(config.pattern_spec) if config.pattern_spec else None
        )
        self.attribute_by_name = {attr.name: attr for attr in config.attributes}

    def generate_full_table(self) -> pd.DataFrame:
        """
        Generate the complete table (all rows across all batches).
        
        Returns:
            DataFrame with all generated rows
        """
        rows = []

        for row_id in range(1, self.config.total_rows + 1):
            row = self._generate_row(row_id)
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def _generate_row(self, row_id: int) -> Dict[str, Any]:
        """
        Generate a single row with all attributes.
        
        Args:
            row_id: Sequential row ID
        
        Returns:
            Dictionary representing one row
        """
        row = {}

        for attr in self.config.attributes:
            if attr.name == self.config.id_column_name:
                row[attr.name] = row_id
            elif attr.name == self.config.timestamp_column_name:
                # Generate timestamp based on pattern windows
                # Each pattern window contains rows_per_window rows
                window_idx = (row_id - 1) // self.config.rows_per_window
                pos_in_window = (row_id - 1) % self.config.rows_per_window
                
                # Distribute rows evenly within their pattern window
                window_start = window_idx * self.config.pattern_window_size
                window_end = (window_idx + 1) * self.config.pattern_window_size
                
                # Uniform distribution within window
                base_time = window_start + (pos_in_window / self.config.rows_per_window) * self.config.pattern_window_size
                # Add small jitter to simulate realistic data
                jitter = self.rng.uniform(-0.1, 0.1)
                row[attr.name] = max(0, base_time + jitter)
            elif attr.attr_type == AttributeType.NUMERICAL:
                row[attr.name] = self._generate_numerical_attribute(attr)
            elif attr.attr_type == AttributeType.CATEGORICAL:
                row[attr.name] = self._generate_categorical_attribute(attr)

        return row

    def _generate_numerical_attribute(self, attr) -> float:
        """Generate value for numerical attribute."""
        if attr.distribution == DistributionType.UNIFORM:
            return self.rng.uniform(attr.min_value, attr.max_value)
        elif attr.distribution == DistributionType.NORMAL:
            mean = (attr.min_value + attr.max_value) / 2
            std = (attr.max_value - attr.min_value) / 6  # 3-sigma rule
            value = self.rng.gauss(mean, std)
            return max(attr.min_value, min(attr.max_value, value))
        elif attr.distribution == DistributionType.ZIPF:
            # Zipfian distribution (skewed towards lower values)
            # Approximate using exponential
            value = self.rng.expovariate(0.1)
            range_size = attr.max_value - attr.min_value
            return attr.min_value + (value % range_size)
        else:
            return self.rng.uniform(attr.min_value, attr.max_value)

    def _generate_categorical_attribute(self, attr) -> str:
        """Generate value for categorical attribute."""
        if not attr.categories:
            return "default"
        return self.rng.choice(attr.categories)

    def _value_for_numerical_condition(self, attr, operator, condition_value, current_value):
        """Generate a numerical value satisfying the comparison operator."""
        try:
            threshold = float(condition_value)
        except (TypeError, ValueError):
            return current_value

        min_v = attr.min_value if attr.min_value is not None else threshold - 100
        max_v = attr.max_value if attr.max_value is not None else threshold + 100

        if operator.value == "=":
            return max(min_v, min(max_v, threshold))
        if operator.value == "<>":
            candidate = self.rng.uniform(min_v, max_v)
            if candidate == threshold:
                candidate = min_v if threshold != min_v else max_v
            return candidate
        if operator.value == "<":
            upper = min(max_v, threshold - 1e-6)
            return min_v if upper <= min_v else self.rng.uniform(min_v, upper)
        if operator.value == "<=":
            upper = min(max_v, threshold)
            return min_v if upper <= min_v else self.rng.uniform(min_v, upper)
        if operator.value == ">":
            lower = max(min_v, threshold + 1e-6)
            return max_v if lower >= max_v else self.rng.uniform(lower, max_v)
        if operator.value == ">=":
            lower = max(min_v, threshold)
            return max_v if lower >= max_v else self.rng.uniform(lower, max_v)

        return current_value

    def _value_for_categorical_condition(self, attr, operator, condition_value, current_value):
        """Generate a categorical value satisfying the comparison operator."""
        categories = attr.categories or []
        if not categories:
            return current_value

        target = str(condition_value)

        if operator.value == "=":
            return target if target in categories else self.rng.choice(categories)

        if operator.value == "<>":
            candidates = [c for c in categories if c != target]
            return self.rng.choice(candidates) if candidates else current_value

        if operator.value == "<":
            candidates = [c for c in categories if c < target]
            return self.rng.choice(candidates) if candidates else current_value

        if operator.value == "<=":
            candidates = [c for c in categories if c <= target]
            return self.rng.choice(candidates) if candidates else current_value

        if operator.value == ">":
            candidates = [c for c in categories if c > target]
            return self.rng.choice(candidates) if candidates else current_value

        if operator.value == ">=":
            candidates = [c for c in categories if c >= target]
            return self.rng.choice(candidates) if candidates else current_value

        return current_value

    def _value_satisfying_condition(self, attr, operator, condition_value, current_value):
        """Generate a value for an attribute that satisfies an independent condition."""
        if attr.attr_type == AttributeType.CATEGORICAL:
            return self._value_for_categorical_condition(
                attr, operator, condition_value, current_value
            )

        return self._value_for_numerical_condition(
            attr, operator, condition_value, current_value
        )

    def apply_pattern_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply PATTERN constraints to the generated table.
        
        This is a placeholder for complex pattern-based data generation.
        Full implementation would:
        - Identify rows matching pattern structure
        - Ensure specific pattern occurrences
        - Adjust target_num_matches if specified
        
        Args:
            df: Generated DataFrame
        
        Returns:
            DataFrame with pattern constraints applied
        """
        if not self.pattern_matcher:
            return df

        # TODO: Implement sophisticated pattern-based adjustment
        # For now, return dataframe as-is
        # In a full implementation, this would:
        # 1. Identify matches of the pattern
        # 2. Adjust data to ensure target_num_matches
        # 3. Preserve pattern structure across batches

        return df

    def apply_define_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply DEFINE clause constraints to the generated table.
        
        This adjusts row values to satisfy selectivity probabilities
        specified in conditions.
        
        Args:
            df: Generated DataFrame
        
        Returns:
            DataFrame with DEFINE constraints applied
        """
        if not self.config.define_spec:
            return df

        define_spec = self.config.define_spec
        df_copy = df.copy()

        # Apply independent conditions
        for condition in define_spec.independent_conditions:
            # Evaluate selectivity for every row in the table.
            indices = range(len(df_copy))
            attr_name = condition.attribute_name

            if attr_name not in df_copy.columns:
                continue

            attr_cfg = self.attribute_by_name.get(attr_name)
            if not attr_cfg:
                continue

            for idx in indices:
                if self.rng.random() < condition.selectivity:
                    current_value = df_copy.loc[idx, attr_name]
                    df_copy.loc[idx, attr_name] = self._value_satisfying_condition(
                        attr_cfg,
                        condition.operator,
                        condition.value,
                        current_value,
                    )

        return df_copy

    def slice_into_batches(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """
        Slice the full table into per-batch DataFrames.
        
        Batches are created according to batch_sizes configuration,
        respecting rows_per_window constraints.
        
        Args:
            df: Full generated DataFrame
        
        Returns:
            Tuple containing initial table DataFrame and list of batch DataFrames
        """
        batches = []
        current_idx = self.config.initial_table_size
        initial_table = df.iloc[:current_idx].copy()

        for batch_size in self.config.batch_sizes:
            end_idx = current_idx + batch_size
            batch_df = df.iloc[current_idx:end_idx].copy()
            batches.append(batch_df)
            current_idx = end_idx

        return initial_table, batches

    def generate(self):
        """
        Orchestrate full generation process.
        
        Workflow:
        1. Generate full table
        2. Apply pattern constraints
        3. Apply DEFINE constraints
        4. Slice into batches
        5. Output to files
        """
        print(f"Generating {self.config.total_rows} rows across {len(self.config.batch_sizes)} batches...")

        # Step 1: Generate full table
        df = self.generate_full_table()
        print(f"✓ Generated full table: {len(df)} rows × {len(df.columns)} columns")

        # Step 2: Apply pattern constraints
        if self.pattern_matcher:
            df = self.apply_pattern_constraints(df)
            print(f"✓ Applied pattern constraints")

        # Step 3: Apply DEFINE constraints
        if self.config.define_spec:
            df = self.apply_define_constraints(df)
            print(f"✓ Applied DEFINE constraints")
            # TODO DependentConditions need to be applied as well

        # Step 4: Slice into batches
        initial_table, batches = self.slice_into_batches(df)
        print(f"✓ Sliced into initial table & {len(batches)} batches")

        # Step 5: Output files
        from .output import OutputWriter
        writer = OutputWriter(self.config)
        writer.write_initial_table(initial_table)
        writer.write_batches(batches)

        print(f"✓ Generation complete!")
        return df, batches
