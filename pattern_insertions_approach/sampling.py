from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ColumnConfig,
    ColumnType,
    ComparisonOperator,
    DependentDistanceCondition,
    Rule,
    RuleType,
)


class ColumnSampler:
    """Samples column values either from rules or default column domains."""

    def __init__(self, columns: List[ColumnConfig], rng: random.Random):
        self.columns: Dict[str, ColumnConfig] = {c.name: c for c in columns}
        self.rng = rng

    def sample_uniform(self, column_name: str) -> Any:
        col = self.columns[column_name]
        if col.column_type == ColumnType.NUMERICAL:
            return self.rng.uniform(col.min_value, col.max_value)
        return self.rng.choice(col.categories)

    def sample_from_rules(self, column_name: str, rules: List[Rule]) -> Any:
        weights = [r.probability for r in rules]
        rule = self.rng.choices(rules, weights=weights, k=1)[0]
        return self._sample_rule(rule)

    def _sample_rule(self, rule: Rule) -> Any:
        if rule.type == RuleType.EXACT:
            return rule.value
        return self.rng.uniform(rule.min, rule.max)


def _clamp_range(start: float, end: float, base_min: float, base_max: float) -> Optional[Tuple[float, float]]:
    s = max(start, base_min)
    e = min(end, base_max)
    if s > e:
        return None
    return s, e


class DependentSampler:
    """Implements constrained-space B sampling with density-preserving weights."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def get_constrained_b(
        self,
        val_a: float,
        rules_b: List[Rule],
        condition: DependentDistanceCondition,
    ) -> Optional[float]:
        # For <= threshold: allowed zone around A.
        # For >= threshold: forbidden zone around A.
        threshold = condition.threshold
        lower = val_a - threshold
        upper = val_a + threshold

        valid_segments: List[Tuple[float, float, float, bool]] = []

        for rule in rules_b:
            if rule.type == RuleType.EXACT:
                value = float(rule.value)
                if self._exact_satisfies(value, lower, upper, condition.operator):
                    valid_segments.append((value, value, rule.probability, True))
                continue

            # Range rule
            r_min = float(rule.min)
            r_max = float(rule.max)
            width = r_max - r_min
            if width < 0:
                continue

            if condition.operator == ComparisonOperator.LTE:
                clipped = _clamp_range(lower, upper, r_min, r_max)
                if clipped is not None:
                    start, end = clipped
                    eff_p = rule.probability if width == 0 else rule.probability * ((end - start) / width)
                    if eff_p > 0:
                        valid_segments.append((start, end, eff_p, False))
            else:
                # >= threshold: [r_min, lower] U [upper, r_max]
                left = _clamp_range(r_min, lower, r_min, r_max)
                right = _clamp_range(upper, r_max, r_min, r_max)
                for clipped in (left, right):
                    if clipped is None:
                        continue
                    start, end = clipped
                    if start > end:
                        continue
                    eff_p = rule.probability if width == 0 else rule.probability * ((end - start) / width)
                    if eff_p > 0:
                        valid_segments.append((start, end, eff_p, False))

        if not valid_segments:
            return None

        weights = [seg[2] for seg in valid_segments]
        if sum(weights) <= 0:
            return None

        start, end, _, is_exact = self.rng.choices(valid_segments, weights=weights, k=1)[0]
        if is_exact or start == end:
            return start
        return self.rng.uniform(start, end)

    @staticmethod
    def _exact_satisfies(value: float, lower: float, upper: float, op: ComparisonOperator) -> bool:
        if op == ComparisonOperator.LTE:
            return lower <= value <= upper
        return value <= lower or value >= upper
