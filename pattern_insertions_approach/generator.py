from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import (
    ComparisonOperator,
    DefineConfig,
    DependentDistanceCondition,
    GapConfig,
    IndependentCondition,
    InsertionPattern,
    OutputConfig,
    RunConfig,
    VariableConfig,
)
from .sampling import ColumnSampler, DependentSampler


@dataclass
class ShortfallRecord:
    batch_index: int
    insertion_index: int
    condition: DependentDistanceCondition
    reason: str


@dataclass
class GenerationReport:
    total_rows: int
    total_insertions_requested: int
    total_insertions_generated: int
    dependent_shortfalls: List[ShortfallRecord] = field(default_factory=list)


class PatternInsertionGenerator:
    """Insertion-first generator based on PATTERN and DEFINE objects."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.column_sampler = ColumnSampler(config.columns, self.rng)
        self.dependent_sampler = DependentSampler(self.rng)

        self._next_id = 1
        self._row_index = 0

    def generate(self) -> tuple[pd.DataFrame, List[pd.DataFrame], GenerationReport]:
        batch_frames: List[pd.DataFrame] = []
        shortfalls: List[ShortfallRecord] = []
        total_insertions_requested = sum(self.config.batch_plan.insertions_per_batch)

        for batch_idx, insertion_count in enumerate(self.config.batch_plan.insertions_per_batch, start=1):
            batch_rows: List[Dict[str, Any]] = []
            for insertion_idx in range(1, insertion_count + 1):
                rows, insertion_shortfalls = self._generate_insertion_rows(batch_idx, insertion_idx)
                batch_rows.extend(rows)
                shortfalls.extend(insertion_shortfalls)

            batch_df = pd.DataFrame(batch_rows)
            if not batch_df.empty:
                batch_df = self._enforce_column_order(batch_df)
            batch_frames.append(batch_df)

        full_df = pd.concat(batch_frames, ignore_index=True) if batch_frames else pd.DataFrame()
        report = GenerationReport(
            total_rows=len(full_df),
            total_insertions_requested=total_insertions_requested,
            total_insertions_generated=total_insertions_requested,
            dependent_shortfalls=shortfalls,
        )
        return full_df, batch_frames, report

    def _generate_insertion_rows(
        self,
        batch_idx: int,
        insertion_idx: int,
    ) -> tuple[List[Dict[str, Any]], List[ShortfallRecord]]:
        logical_rows: List[Dict[str, Any]] = []
        shortfalls: List[ShortfallRecord] = []

        logical_rows.extend(self._generate_filler_rows(self.config.space.before_first_variable))

        var_blocks = self._generate_variable_blocks()
        dep_shortfalls = self._enforce_dependent_conditions(var_blocks, batch_idx, insertion_idx)
        shortfalls.extend(dep_shortfalls)

        for i, var_name in enumerate(self.config.pattern.variable_order):
            logical_rows.extend(var_blocks[var_name])
            if i < len(self.config.pattern.variable_order) - 1:
                gap_len = self._sample_gap_length(i)
                logical_rows.extend(self._generate_filler_rows(gap_len))

        logical_rows.extend(self._generate_filler_rows(self.config.space.after_last_variable))

        rows = [self._stamp_base_fields(row) for row in logical_rows]
        return rows, shortfalls

    def _generate_variable_blocks(self) -> Dict[str, List[Dict[str, Any]]]:
        blocks: Dict[str, List[Dict[str, Any]]] = {}
        for var_name in self.config.pattern.variable_order:
            variable = self.config.variables[var_name]
            multiplicity = self._sample_variable_multiplicity(variable)
            block_rows = [self._generate_variable_row(variable) for _ in range(multiplicity)]
            blocks[var_name] = block_rows
        return blocks

    def _sample_variable_multiplicity(self, variable: VariableConfig) -> int:
        if variable.kleene is None:
            return 1
        low, high = variable.kleene.sample_length_bounds()
        return self.rng.randint(low, high)

    def _sample_gap_length(self, gap_index: int) -> int:
        if not self.config.pattern.gaps_between_variables:
            return 0
        gap_cfg: GapConfig = self.config.pattern.gaps_between_variables[gap_index]
        low, high = gap_cfg.sample_length_bounds()
        return self.rng.randint(low, high)

    def _generate_variable_row(self, variable: VariableConfig) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for col in self.config.columns:
            rules = variable.column_distributions.get(col.name)
            if rules:
                row[col.name] = self.column_sampler.sample_from_rules(col.name, rules)
            else:
                row[col.name] = self.column_sampler.sample_uniform(col.name)

        self._apply_independent_conditions(row, variable.name)
        return row

    def _apply_independent_conditions(self, row: Dict[str, Any], variable_name: str) -> None:
        for cond in self.config.define.independent_conditions:
            if cond.variable_name != variable_name:
                continue
            row[cond.column_name] = self._sample_value_satisfying_condition(
                current_value=row.get(cond.column_name),
                cond=cond,
            )

    def _sample_value_satisfying_condition(self, current_value: Any, cond: IndependentCondition) -> Any:
        # Best-effort satisfying rewrite over the domain by rejection from column sampler.
        for _ in range(100):
            candidate = current_value if _ == 0 else self.column_sampler.sample_uniform(cond.column_name)
            if self._compare(candidate, cond.operator, cond.value):
                return candidate
        return current_value

    def _compare(self, left: Any, op: ComparisonOperator, right: Any) -> bool:
        if op == ComparisonOperator.EQ:
            return left == right
        if op == ComparisonOperator.NEQ:
            return left != right
        if op == ComparisonOperator.LT:
            return left < right
        if op == ComparisonOperator.LTE:
            return left <= right
        if op == ComparisonOperator.GT:
            return left > right
        if op == ComparisonOperator.GTE:
            return left >= right
        return False

    def _enforce_dependent_conditions(
        self,
        var_blocks: Dict[str, List[Dict[str, Any]]],
        batch_idx: int,
        insertion_idx: int,
    ) -> List[ShortfallRecord]:
        shortfalls: List[ShortfallRecord] = []
        for cond in self.config.define.dependent_conditions:
            rows_a = var_blocks.get(cond.var_a, [])
            rows_b = var_blocks.get(cond.var_b, [])
            if not rows_a or not rows_b:
                shortfalls.append(
                    ShortfallRecord(
                        batch_index=batch_idx,
                        insertion_index=insertion_idx,
                        condition=cond,
                        reason="missing variable rows for condition",
                    )
                )
                continue

            # We enforce on the first occurrence of each variable for this insertion.
            row_a = rows_a[0]
            row_b = rows_b[0]

            if self._dependent_holds(row_a, row_b, cond):
                continue

            success = self._repair_dependent(row_a, row_b, cond)
            if not success:
                shortfalls.append(
                    ShortfallRecord(
                        batch_index=batch_idx,
                        insertion_index=insertion_idx,
                        condition=cond,
                        reason="retry budget exhausted",
                    )
                )

        return shortfalls

    def _repair_dependent(self, row_a: Dict[str, Any], row_b: Dict[str, Any], cond: DependentDistanceCondition) -> bool:
        var_a_cfg = self.config.variables[cond.var_a]
        var_b_cfg = self.config.variables[cond.var_b]
        rules_a = var_a_cfg.column_distributions.get(cond.column_name)
        rules_b = var_b_cfg.column_distributions.get(cond.column_name)

        # If no explicit rules exist, fall back to rejection over uniform draws.
        if not rules_a or not rules_b:
            for _ in range(cond.max_retries):
                row_a[cond.column_name] = self.column_sampler.sample_uniform(cond.column_name)
                row_b[cond.column_name] = self.column_sampler.sample_uniform(cond.column_name)
                if self._dependent_holds(row_a, row_b, cond):
                    return True
            return False

        for _ in range(cond.max_retries):
            val_a = self.column_sampler.sample_from_rules(cond.column_name, rules_a)
            val_b = self.dependent_sampler.get_constrained_b(val_a, rules_b, cond)
            if val_b is None:
                continue
            row_a[cond.column_name] = val_a
            row_b[cond.column_name] = val_b
            if self._dependent_holds(row_a, row_b, cond):
                return True

        return False

    def _dependent_holds(
        self,
        row_a: Dict[str, Any],
        row_b: Dict[str, Any],
        cond: DependentDistanceCondition,
    ) -> bool:
        distance = abs(float(row_a[cond.column_name]) - float(row_b[cond.column_name]))
        if cond.operator == ComparisonOperator.LTE:
            return distance <= cond.threshold
        return distance >= cond.threshold

    def _generate_filler_rows(self, count: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _ in range(count):
            row: Dict[str, Any] = {}
            for col in self.config.columns:
                row[col.name] = self.column_sampler.sample_uniform(col.name)
            rows.append(row)
        return rows

    def _stamp_base_fields(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row_id = self._next_id
        self._next_id += 1

        window_idx = self._row_index // self.config.rows_per_window
        pos_in_window = self._row_index % self.config.rows_per_window
        self._row_index += 1

        window_start = window_idx * self.config.pattern_window_size
        timestamp = window_start + (pos_in_window / self.config.rows_per_window) * self.config.pattern_window_size

        return {
            self.config.id_column_name: row_id,
            self.config.timestamp_column_name: timestamp,
            **row,
        }

    def _enforce_column_order(self, df: pd.DataFrame) -> pd.DataFrame:
        ordered = [self.config.id_column_name, self.config.timestamp_column_name] + [c.name for c in self.config.columns]
        return df[ordered]

    def write_outputs(self, batches: List[pd.DataFrame], output_config: Optional[OutputConfig] = None) -> None:
        out_cfg = output_config or self.config.output

        # Reuse existing writer to keep parquet/csv/sql behavior identical.
        from match_recognize_datagen.output import OutputWriter

        class _WriterCompatConfig:
            output_dir = out_cfg.output_dir
            output_format = out_cfg.output_format

        writer = OutputWriter(_WriterCompatConfig())
        for idx, batch_df in enumerate(batches):
            writer._write_batch(batch_df, idx + 1)
