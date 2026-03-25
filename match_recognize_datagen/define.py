"""
DEFINE clause handling and condition evaluation for MATCH RECOGNIZE queries.
"""

from typing import Dict, Any, Callable, List
from .config import (
    DefineSpec,
    IndependentCondition,
    DependentConditionPair,
    WindowCondition,
    ComparisonOperator,
)


class ConditionEvaluator:
    """Evaluator for DEFINE clause conditions."""

    # Mapping of operators to comparison functions
    OPERATORS = {
        ComparisonOperator.EQ: lambda a, b: a == b,
        ComparisonOperator.NEQ: lambda a, b: a != b,
        ComparisonOperator.LT: lambda a, b: a < b,
        ComparisonOperator.LTE: lambda a, b: a <= b,
        ComparisonOperator.GT: lambda a, b: a > b,
        ComparisonOperator.GTE: lambda a, b: a >= b,
    }

    @staticmethod
    def evaluate_independent_condition(
        row: Dict[str, Any],
        condition: IndependentCondition,
    ) -> bool:
        """
        Evaluate an independent condition on a single row.
        
        Args:
            row: Data row as dictionary
            condition: IndependentCondition to evaluate
        
        Returns:
            True if condition holds, False otherwise
        """
        if condition.attribute_name not in row:
            raise ValueError(f"Attribute '{condition.attribute_name}' not in row")

        row_value = row[condition.attribute_name]
        op_func = ConditionEvaluator.OPERATORS[condition.operator]
        return op_func(row_value, condition.value)

    @staticmethod
    def evaluate_pairwise_condition(
        row1: Dict[str, Any],
        row2: Dict[str, Any],
        condition: DependentConditionPair,
    ) -> bool:
        """
        Evaluate a range/dependent condition on two rows.
        
        Examples:
        - Vi.x > Vj.x - 0.05 AND Vi.x < Vj.x + 0.05
        - General pairwise comparisons
        
        Args:
            row1: Data row for first variable
            row2: Data row for second variable
            condition: DependentConditionPair to evaluate
        
        Returns:
            True if condition holds, False otherwise
        """
        if condition.var1_attr not in row1:
            raise ValueError(f"Attribute '{condition.var1_attr}' not in first row")
        if condition.var2_attr not in row2:
            raise ValueError(f"Attribute '{condition.var2_attr}' not in second row")

        val1 = row1[condition.var1_attr]
        val2 = row2[condition.var2_attr]

        # For range predicates with threshold
        if condition.threshold is not None:
            if condition.operator == ComparisonOperator.LT:
                # val1 < val2 + threshold
                return val1 < val2 + condition.threshold
            elif condition.operator == ComparisonOperator.GT:
                # val1 > val2 - threshold
                return val1 > val2 - condition.threshold
            elif condition.operator == ComparisonOperator.LTE:
                return val1 <= val2 + condition.threshold
            elif condition.operator == ComparisonOperator.GTE:
                return val1 >= val2 - condition.threshold

        # Direct comparison
        op_func = ConditionEvaluator.OPERATORS[condition.operator]
        return op_func(val1, val2)

    @staticmethod
    def evaluate_window_condition(
        row1: Dict[str, Any],
        row2: Dict[str, Any],
        condition: WindowCondition,
    ) -> bool:
        """
        Evaluate a window condition between two temporal events.
        
        Example: Vn.t - V1.t < 30 Mins
        
        Args:
            row1: Data row for first variable (typically earlier in time)
            row2: Data row for second variable (typically later in time)
            condition: WindowCondition to evaluate
        
        Returns:
            True if condition holds, False otherwise
        """
        if condition.time_attr not in row1 or condition.time_attr not in row2:
            raise ValueError(f"Time attribute '{condition.time_attr}' not in rows")

        time1 = row1[condition.time_attr]
        time2 = row2[condition.time_attr]
        time_diff = abs(time2 - time1)

        if condition.max_time_diff is None:
            return True

        return time_diff <= condition.max_time_diff


class SelectivityApplier:
    """Apply selectivity probabilities to conditions."""

    @staticmethod
    def apply_independent_selectivity(
        row: Dict[str, Any],
        condition: IndependentCondition,
        rng=None,
    ) -> bool:
        """
        Apply independent condition with selectivity.
        
        With probability selectivity, condition is satisfied.
        With probability (1 - selectivity), condition is not satisfied.
        """
        if rng is None:
            import random
            rng = random

        # Determine if we should satisfy condition based on selectivity
        if rng.random() > condition.selectivity:
            return False

        # If we reach here, condition should be satisfied
        # Adjust row value to satisfy condition
        row_copy = row.copy()
        row_copy[condition.attribute_name] = condition.value #TODO das ist nicht korrekt einfach auf den condition.value zu setzen bei e.g. < still not satisfied
        
        return ConditionEvaluator.evaluate_independent_condition(row_copy, condition)

    @staticmethod
    def get_selectivity(condition: Any) -> float:
        """Extract selectivity value from any condition type."""
        if hasattr(condition, 'selectivity'):
            return condition.selectivity
        return 0.5  # Default
