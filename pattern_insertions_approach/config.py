from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class ColumnType(Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class RuleType(Enum):
    EXACT = "exact"
    RANGE = "range"


class ComparisonOperator(Enum):
    EQ = "="
    NEQ = "<>"
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="


@dataclass
class Rule:
    """Weighted sampling rule for a column distribution config.

    EXACT: uses `value`.
    RANGE: samples uniformly in [min, max].
    """

    type: RuleType
    probability: float
    value: Optional[Any] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def __post_init__(self):
        if self.probability < 0:
            raise ValueError("Rule probability must be non-negative")
        if self.type == RuleType.EXACT and self.value is None:
            raise ValueError("Exact rule requires value")
        if self.type == RuleType.RANGE:
            if self.min is None or self.max is None:
                raise ValueError("Range rule requires min and max")
            if self.min > self.max:
                raise ValueError("Range rule min must be <= max")


@dataclass
class ColumnConfig:
    name: str
    column_type: ColumnType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[Any]] = None

    def __post_init__(self):
        if self.column_type == ColumnType.NUMERICAL:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"Numerical column {self.name} requires min_value/max_value")
            if self.min_value > self.max_value:
                raise ValueError(f"Numerical column {self.name} has min_value > max_value")
        if self.column_type == ColumnType.CATEGORICAL:
            if not self.categories:
                raise ValueError(f"Categorical column {self.name} requires non-empty categories")


@dataclass
class KleeneConfig:
    fixed_length: Optional[int] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    def sample_length_bounds(self) -> tuple[int, int]:
        if self.fixed_length is not None:
            if self.fixed_length <= 0:
                raise ValueError("Kleene fixed_length must be > 0")
            return self.fixed_length, self.fixed_length
        if self.min_length is None or self.max_length is None:
            raise ValueError("Kleene config requires fixed_length or min_length/max_length")
        if self.min_length <= 0 or self.max_length <= 0:
            raise ValueError("Kleene min/max length must be > 0")
        if self.min_length > self.max_length:
            raise ValueError("Kleene min_length must be <= max_length")
        return self.min_length, self.max_length


@dataclass
class VariableConfig:
    """Variable with optional per-column weighted rule configs."""

    name: str
    kleene: Optional[KleeneConfig] = None
    column_distributions: Dict[str, List[Rule]] = field(default_factory=dict)


@dataclass
class GapConfig:
    """Gap (Z*) between consecutive pattern variables."""

    fixed_length: Optional[int] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    def sample_length_bounds(self) -> tuple[int, int]:
        if self.fixed_length is not None:
            if self.fixed_length < 0:
                raise ValueError("Gap fixed_length must be >= 0")
            return self.fixed_length, self.fixed_length
        if self.min_length is None or self.max_length is None:
            raise ValueError("Gap requires fixed_length or min_length/max_length")
        if self.min_length < 0 or self.max_length < 0:
            raise ValueError("Gap min/max length must be >= 0")
        if self.min_length > self.max_length:
            raise ValueError("Gap min_length must be <= max_length")
        return self.min_length, self.max_length


@dataclass
class SpaceConfig:
    """Filler row count around each insertion."""

    before_first_variable: int = 0
    after_last_variable: int = 0

    def __post_init__(self):
        if self.before_first_variable < 0 or self.after_last_variable < 0:
            raise ValueError("SpaceConfig values must be >= 0")


@dataclass
class InsertionPattern:
    """Pattern made of variable names in order and Z* gaps between them."""

    variable_order: List[str]
    gaps_between_variables: List[GapConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.variable_order:
            raise ValueError("Pattern variable_order must be non-empty")
        if self.gaps_between_variables and len(self.gaps_between_variables) != len(self.variable_order) - 1:
            raise ValueError("Pattern gaps_between_variables must have len(variable_order)-1 elements")


@dataclass
class IndependentCondition:
    variable_name: str
    column_name: str
    operator: ComparisonOperator
    value: Any


@dataclass
class DependentDistanceCondition:
    """Constraint between two variables over one numeric column.

    Uses abs(A - B) with operator <= threshold or >= threshold.
    """

    var_a: str
    var_b: str
    column_name: str
    operator: Literal[ComparisonOperator.LTE, ComparisonOperator.GTE]
    threshold: float
    max_retries: int = 100

    def __post_init__(self):
        if self.operator not in (ComparisonOperator.LTE, ComparisonOperator.GTE):
            raise ValueError("Dependent operator must be <= or >=")
        if self.threshold < 0:
            raise ValueError("Dependent threshold must be >= 0")
        if self.max_retries <= 0:
            raise ValueError("Dependent max_retries must be > 0")


@dataclass
class DefineConfig:
    independent_conditions: List[IndependentCondition] = field(default_factory=list)
    dependent_conditions: List[DependentDistanceCondition] = field(default_factory=list)


@dataclass
class BatchPlan:
    """Number of pattern insertions per batch."""

    insertions_per_batch: List[int]

    def __post_init__(self):
        if not self.insertions_per_batch:
            raise ValueError("insertions_per_batch must be non-empty")
        if any(v < 0 for v in self.insertions_per_batch):
            raise ValueError("insertions_per_batch values must be >= 0")


@dataclass
class OutputConfig:
    output_dir: str
    output_format: Literal["parquet", "csv", "sql"] = "parquet"


@dataclass
class RunConfig:
    columns: List[ColumnConfig]
    variables: Dict[str, VariableConfig]
    pattern: InsertionPattern
    define: DefineConfig
    batch_plan: BatchPlan
    rows_per_window: int
    pattern_window_size: float = 300.0
    space: SpaceConfig = field(default_factory=SpaceConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(output_dir="./output"))
    id_column_name: str = "id"
    timestamp_column_name: str = "timestamp"
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.rows_per_window <= 0:
            raise ValueError("rows_per_window must be > 0")
        if self.pattern_window_size <= 0:
            raise ValueError("pattern_window_size must be > 0")

        col_names = {c.name for c in self.columns}
        col_by_name = {c.name: c for c in self.columns}
        for required in self.pattern.variable_order:
            if required not in self.variables:
                raise ValueError(f"Pattern variable {required} missing in variables map")

        if len(set(self.pattern.variable_order)) != len(self.pattern.variable_order):
            raise ValueError("Pattern variable_order must not contain duplicate variable names")

        for var in self.variables.values():
            for col_name, rules in var.column_distributions.items():
                if col_name not in col_names:
                    raise ValueError(f"Variable {var.name} references unknown column {col_name}")
                if not rules:
                    raise ValueError(f"Variable {var.name} column {col_name} has empty rule list")
                total_prob = sum(r.probability for r in rules)
                if total_prob <= 0:
                    raise ValueError(f"Variable {var.name} column {col_name} has non-positive total rule probability")
                column_cfg = col_by_name[col_name]
                for rule in rules:
                    if column_cfg.column_type == ColumnType.CATEGORICAL and rule.type == RuleType.RANGE:
                        raise ValueError(
                            f"Variable {var.name} column {col_name} cannot use range rules for categorical columns"
                        )
                    if column_cfg.column_type == ColumnType.NUMERICAL and rule.type == RuleType.EXACT:
                        if not isinstance(rule.value, (int, float)):
                            raise ValueError(
                                f"Variable {var.name} column {col_name} exact rule must be numeric"
                            )

        for cond in self.define.independent_conditions:
            if cond.variable_name not in self.variables:
                raise ValueError(f"Independent condition references unknown variable {cond.variable_name}")
            if cond.column_name not in col_names:
                raise ValueError(f"Independent condition references unknown column {cond.column_name}")

        for cond in self.define.dependent_conditions:
            if cond.var_a not in self.variables or cond.var_b not in self.variables:
                raise ValueError(
                    f"Dependent condition references unknown variable pair {cond.var_a}, {cond.var_b}"
                )
            if cond.column_name not in col_names:
                raise ValueError(f"Dependent condition references unknown column {cond.column_name}")
            if col_by_name[cond.column_name].column_type != ColumnType.NUMERICAL:
                raise ValueError("Dependent distance conditions require a numerical column")
