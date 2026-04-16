"""
MATCH RECOGNIZE Synthetic Data Generator

A Python library for generating synthetic data for MATCH RECOGNIZE queries
with configurable row-window generation, constraint definitions, and
streaming batch output.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .config import (
    GeneratorConfig,
    AttributeConfig,
    DefineSpec,
    IndependentCondition,
    DependentConditionPair,
    ComparisonOperator,
    AttributeType,
    DistributionType,
)
from .generator import DataGenerator
from .define import DefineConstraintApplier
from .output import OutputWriter

__all__ = [
    "GeneratorConfig",
    "AttributeConfig",
    "DefineSpec",
    "IndependentCondition",
    "DependentConditionPair",
    "ComparisonOperator",
    "AttributeType",
    "DistributionType",
    "DataGenerator",
    "DefineConstraintApplier",
    "OutputWriter",
]
