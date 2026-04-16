"""
Configuration and data models for the MATCH RECOGNIZE synthetic data generator.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Any
from enum import Enum


class DistributionType(Enum):
    """Supported variable distributions."""
    UNIFORM = "uniform"
    ZIPF = "zipf"
    NORMAL = "normal"


class AttributeType(Enum):
    """Attribute types for table columns."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class ComparisonOperator(Enum):
    """Supported comparison operators in DEFINE clause."""
    EQ = "="
    NEQ = "<>"
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="


@dataclass
class AttributeConfig:
    """Configuration for a single table attribute."""
    name: str
    attr_type: AttributeType
    # For numerical attributes
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    distribution: DistributionType = DistributionType.UNIFORM
    # For categorical attributes
    categories: Optional[List[str]] = None


@dataclass
class IndependentCondition:
    """Independent condition per variable (e.g., Vi.val > 42)."""
    variable_name: str
    attribute_name: str
    operator: ComparisonOperator
    value: Any
    selectivity: float = 0.5  # Probability that condition holds


@dataclass
class DependentConditionPair:
    """Dependent condition for pair of variables."""
    var1_name: str
    var2_name: str
    var1_attr: str
    var2_attr: str
    operator: ComparisonOperator
    threshold: Optional[float] = None  # For range predicates
    selectivity: float = 0.5


@dataclass
class DefineSpec:
    """MATCH RECOGNIZE DEFINE clause specification."""
    independent_conditions: List[IndependentCondition] = field(default_factory=list)
    pairwise_conditions: List[DependentConditionPair] = field(default_factory=list)


@dataclass
class GeneratorConfig:
    """Main configuration for the synthetic data generator."""
    # Table dimensions
    initial_table_size: int
    total_rows: int
    num_columns: int

    # Batching
    batch_sizes: List[int]
    rows_per_window: int

    # Temporal window configuration for row timestamp spread
    pattern_window_size: float = 300.0  # In seconds (default 5 minutes)

    # Attributes
    attributes: List[AttributeConfig] = field(default_factory=list)
    id_column_name: str = "id"
    timestamp_column_name: str = "timestamp"

    # Constraints
    define_spec: Optional[DefineSpec] = None

    # Output
    output_format: Literal["parquet", "csv", "sql"] = "parquet"
    output_dir: str = "./output"

    def __post_init__(self):
        """Validation and post-processing."""
        if sum(self.batch_sizes) + self.initial_table_size != self.total_rows:
            raise ValueError(
                f"Batch sizes sum ({sum(self.batch_sizes)}) + initial_table_size "
                f"({self.initial_table_size}) must equal total_rows ({self.total_rows})"
            )

        if self.rows_per_window <= 0:
            raise ValueError("rows_per_window must be positive")

        if self.pattern_window_size <= 0:
            raise ValueError("pattern_window_size must be positive")

        if self.total_rows % self.rows_per_window != 0:
            raise ValueError(
                f"total_rows ({self.total_rows}) must be divisible by rows_per_window ({self.rows_per_window})"
            )

        if self.num_columns < 3:  # ID, timestamp, + at least 1 other
            raise ValueError("num_columns must be at least 3 (ID, timestamp, + others)")

        # Ensure ID and timestamp columns are in attributes
        if not any(attr.name == self.id_column_name for attr in self.attributes):
            self.attributes.insert(0, AttributeConfig(
                name=self.id_column_name,
                attr_type=AttributeType.NUMERICAL,
                min_value=1,
                max_value=self.total_rows
            ))

        if not any(attr.name == self.timestamp_column_name for attr in self.attributes):
            self.attributes.insert(1, AttributeConfig(
                name=self.timestamp_column_name,
                attr_type=AttributeType.NUMERICAL,
                min_value=0,
                max_value=86400  # 1 day in seconds
            ))
