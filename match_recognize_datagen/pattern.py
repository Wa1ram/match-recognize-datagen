"""
PATTERN clause handling and validation for MATCH RECOGNIZE queries.
"""

from typing import List, Dict, Tuple
from .config import PatternSpec, PatternElement, KleeneConfig


class PatternParser:
    """Parser and validator for PATTERN specifications."""

    @staticmethod
    def validate_pattern(pattern_spec: PatternSpec) -> bool:
        """
        Validate PATTERN specification.
        
        Rules:
        - At least one variable
        - Variables are unique
        - Kleene+ configs reference valid variables
        """
        if not pattern_spec.elements:
            raise ValueError("Pattern must contain at least one element")

        variable_names = [elem.variable_name for elem in pattern_spec.elements]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("Variable names must be unique in pattern")

        for var_name, kleene_cfg in pattern_spec.kleene_configs.items():
            if var_name not in variable_names:
                raise ValueError(f"Kleene+ config references undefined variable: {var_name}")

        return True

    @staticmethod
    def get_variables(pattern_spec: PatternSpec) -> List[str]:
        """Extract ordered list of variables from pattern."""
        return [elem.variable_name for elem in pattern_spec.elements]

    @staticmethod
    def get_kleene_variables(pattern_spec: PatternSpec) -> List[str]:
        """Extract variables with Kleene+ operator."""
        return [
            elem.variable_name for elem in pattern_spec.elements
            if elem.is_kleene_plus
        ]

    @staticmethod
    def pattern_to_string(pattern_spec: PatternSpec) -> str:
        """
        Convert pattern to readable string format.
        
        Example: V1 Z* V2+ Z* V3
        Where Z* represents variable-length wildcards (gaps)
        """
        parts = []
        for i, elem in enumerate(pattern_spec.elements):
            parts.append(elem.variable_name)
            if elem.is_kleene_plus:
                parts[-1] += "+"

            # Add wildcard after each element except last
            if i < len(pattern_spec.elements) - 1:
                parts.append("Z*")

        return " ".join(parts)


class PatternMatcher: #TODO use to enable #matches parameter
    """Logic for pattern matching in sequences."""

    def __init__(self, pattern_spec: PatternSpec):
        self.pattern_spec = pattern_spec
        PatternParser.validate_pattern(pattern_spec)
        self.variables = PatternParser.get_variables(pattern_spec)
        self.kleene_vars = PatternParser.get_kleene_variables(pattern_spec)

