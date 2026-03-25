"""
MATCH RECOGNIZE Synthetic Data Generator

A Python library for generating synthetic data for MATCH RECOGNIZE queries
with configurable pattern specifications, constraint definitions, and
streaming batch output.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .config import (
    GeneratorConfig,
    AttributeConfig,
    PatternSpec,
    PatternElement,
    KleeneConfig,
    DefineSpec,
    IndependentCondition,
    DependentConditionPair,
    WindowCondition,
    ComparisonOperator,
    AttributeType,
    DistributionType,
)
from .generator import DataGenerator
from .output import OutputWriter

__all__ = [
    "GeneratorConfig",
    "AttributeConfig",
    "PatternSpec",
    "PatternElement",
    "KleeneConfig",
    "DefineSpec",
    "IndependentCondition",
    "DependentConditionPair",
    "WindowCondition",
    "ComparisonOperator",
    "AttributeType",
    "DistributionType",
    "DataGenerator",
    "PatternParser",
    "PatternMatcher",
    "ConditionEvaluator",
    "SelectivityApplier",
    "OutputWriter",
]
