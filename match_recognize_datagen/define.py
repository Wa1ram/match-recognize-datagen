"""
DEFINE constraint application logic.
"""

from __future__ import annotations

import random
from typing import Any, List

import pandas as pd

from .config import AttributeType, GeneratorConfig


class DefineConstraintApplier:
    """Apply DEFINE independent and pairwise conditions to a dataframe."""

    def __init__(self, config: GeneratorConfig, rng: random.Random):
        self.config = config
        self.rng = rng
        self.attribute_by_name = {attr.name: attr for attr in config.attributes}

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

    def _condition_holds(self, value, operator, condition_value) -> bool:
        """Return whether a value satisfies the condition operator/value."""
        try:
            lhs = float(value)
            rhs = float(condition_value)
        except (TypeError, ValueError):
            lhs = str(value)
            rhs = str(condition_value)

        if operator.value == "=":
            return lhs == rhs
        if operator.value == "<>":
            return lhs != rhs
        if operator.value == "<":
            return lhs < rhs
        if operator.value == "<=":
            return lhs <= rhs
        if operator.value == ">":
            return lhs > rhs
        if operator.value == ">=":
            return lhs >= rhs

        return False

    def _pair_condition_holds(self, left_value, right_value, operator, threshold) -> bool:
        """Return whether a pair condition holds for two values."""
        if threshold is None:
            return self._condition_holds(left_value, operator, right_value)

        try:
            lhs = float(left_value)
            rhs = float(right_value)
            thr = float(threshold)
        except (TypeError, ValueError):
            return False

        distance = abs(lhs - rhs)
        return self._condition_holds(distance, operator, thr)

    @staticmethod
    def _pair_count(k: int) -> int:
        """Number of unordered pairs in a set of size k."""
        return (k * (k - 1)) // 2

    @staticmethod
    def _clamp_selectivity(selectivity: float) -> float:
        return max(0.0, min(1.0, float(selectivity)))

    def _enforce_independent_condition_exact(self, df: pd.DataFrame, condition) -> None:
        """Enforce independent condition with exact row-count target."""
        attr_name = condition.attribute_name
        if attr_name not in df.columns:
            return

        attr_cfg = self.attribute_by_name.get(attr_name)
        if not attr_cfg:
            return

        n_rows = len(df)
        if n_rows == 0:
            return

        target_selectivity = self._clamp_selectivity(condition.selectivity)
        target_count = int(round(target_selectivity * n_rows))

        satisfied_indices = []
        unsatisfied_indices = []
        for idx in range(n_rows):
            if self._condition_holds(df.loc[idx, attr_name], condition.operator, condition.value):
                satisfied_indices.append(idx)
            else:
                unsatisfied_indices.append(idx)

        current_count = len(satisfied_indices)
        if current_count == target_count:
            return

        if current_count < target_count:
            need = min(target_count - current_count, len(unsatisfied_indices))
            for idx in self.rng.sample(unsatisfied_indices, need):
                current_value = df.loc[idx, attr_name]
                df.loc[idx, attr_name] = self._value_satisfying_condition(
                    attr_cfg,
                    condition.operator,
                    condition.value,
                    current_value,
                )
            return

        need = min(current_count - target_count, len(satisfied_indices))
        for idx in self.rng.sample(satisfied_indices, need):
            current_value = df.loc[idx, attr_name]
            df.loc[idx, attr_name] = self._value_violating_condition(
                attr_cfg,
                condition.operator,
                condition.value,
                current_value,
            )

    def _largest_group_size_for_pairs(self, max_size: int, max_pairs: int) -> int:
        """Largest k <= max_size with C(k,2) <= max_pairs."""
        low, high = 0, max_size
        while low < high:
            mid = (low + high + 1) // 2
            if self._pair_count(mid) <= max_pairs:
                low = mid
            else:
                high = mid - 1
        return low

    def _group_sizes_for_target_close_pairs(self, n_rows: int, target_close_pairs: int) -> List[int]:
        """Build group sizes such that sum(C(size,2)) is close to target."""
        groups: List[int] = []
        remaining_rows = n_rows
        remaining_pairs = max(0, target_close_pairs)

        # Phase 1: consume large triangular chunks greedily.
        while remaining_pairs > 0 and remaining_rows > 1:
            k = self._largest_group_size_for_pairs(remaining_rows, remaining_pairs)
            if k < 2:
                break

            groups.append(k)
            remaining_rows -= k
            remaining_pairs -= self._pair_count(k)

        # Phase 2: fine-tune by growing existing groups one row at a time.
        changed = True
        while remaining_pairs > 0 and remaining_rows > 0 and changed:
            changed = False
            for idx, size in enumerate(groups):
                if remaining_rows == 0:
                    break
                increment = size  # C(size+1,2) - C(size,2)
                if increment <= remaining_pairs:
                    groups[idx] = size + 1
                    remaining_rows -= 1
                    remaining_pairs -= increment
                    changed = True

        # Fill all unassigned rows as singletons (zero close pairs).
        if remaining_rows > 0:
            groups.extend([1] * remaining_rows)

        return groups

    def _enforce_pairwise_condition(self, df: pd.DataFrame, condition) -> None:
        """
        Enforce pairwise selectivity using a distance-group construction.

        Construction rationale:
        - rows in the same group get identical values -> distance 0
        - different groups are separated by a gap larger than threshold
        This lets us control the number of close-vs-far unordered row pairs directly
        while avoiding O(n^2) iterative rewrites.
        """
        attr_name = condition.var1_attr
        if condition.var1_attr != condition.var2_attr:
            return
        if condition.threshold is None:
            return
        if attr_name not in df.columns:
            return

        attr_cfg = self.attribute_by_name.get(attr_name)
        if not attr_cfg or attr_cfg.attr_type != AttributeType.NUMERICAL:
            return

        n_rows = len(df)
        if n_rows < 2:
            return

        try:
            threshold = float(condition.threshold)
        except (TypeError, ValueError):
            return

        total_pairs = self._pair_count(n_rows)
        if total_pairs == 0:
            return

        target_selectivity = self._clamp_selectivity(condition.selectivity)
        target_satisfied_pairs = int(round(target_selectivity * total_pairs))

        operator_value = condition.operator.value
        supports_distance_mode = operator_value in {"<", "<=", ">", ">="}
        if not supports_distance_mode:
            return

        # For <,<= : satisfying pairs are "close" pairs (distance <= threshold).
        # For >,>= : satisfying pairs are "far" pairs; close pairs are the complement.
        if operator_value in {"<", "<="}:
            target_close_pairs = target_satisfied_pairs
        else:
            target_close_pairs = total_pairs - target_satisfied_pairs

        target_close_pairs = max(0, min(total_pairs, target_close_pairs))
        groups = self._group_sizes_for_target_close_pairs(n_rows, target_close_pairs)

        # Stable bounds to keep generated values in configured range where possible.
        min_v = attr_cfg.min_value
        max_v = attr_cfg.max_value
        if min_v is None:
            min_v = float(df[attr_name].min()) if n_rows > 0 else 0.0
        if max_v is None:
            max_v = float(df[attr_name].max()) if n_rows > 0 else min_v + 100.0

        gap = max(1.0, (threshold + 1e-6) * 2.0)
        assign_indices = list(range(n_rows))
        self.rng.shuffle(assign_indices)

        cursor = 0
        value_anchor = float(min_v)
        for group_size in groups:
            if cursor >= n_rows:
                break

            end = min(cursor + group_size, n_rows)
            row_ids = assign_indices[cursor:end]

            # Keep anchors in range if possible by wrapping around.
            group_value = value_anchor
            if group_value > max_v:
                group_value = float(min_v) + ((group_value - float(min_v)) % max(gap, 1.0))

            for row_idx in row_ids:
                df.loc[row_idx, attr_name] = group_value

            cursor = end
            value_anchor += gap

    def _value_violating_condition(self, attr, operator, condition_value, current_value):
        """Generate a value that violates the given condition."""
        if attr.attr_type == AttributeType.CATEGORICAL:
            categories = attr.categories or []
            if not categories:
                return current_value

            violating_candidates = [
                c for c in categories
                if not self._condition_holds(c, operator, condition_value)
            ]
            return self.rng.choice(violating_candidates) if violating_candidates else current_value

        try:
            threshold = float(condition_value)
        except (TypeError, ValueError):
            return current_value

        min_v = attr.min_value if attr.min_value is not None else threshold - 100
        max_v = attr.max_value if attr.max_value is not None else threshold + 100

        if operator.value == "=":
            # Simplest valid violation: choose any in-bounds value different from threshold.
            if min_v == max_v:
                return current_value
            for _ in range(50):
                candidate = self.rng.uniform(min_v, max_v)
                if candidate != threshold:
                    return candidate

            # Fallback (extremely unlikely): deterministic in-bounds alternative.
            if threshold != min_v:
                return min_v
            if threshold != max_v:
                return max_v
            return current_value
        if operator.value == "<>":
            return max(min_v, min(max_v, threshold))
        if operator.value == "<":
            lower = max(min_v, threshold)
            return max_v if lower >= max_v else self.rng.uniform(lower, max_v)
        if operator.value == "<=":
            lower = max(min_v, threshold + 1e-6)
            return max_v if lower >= max_v else self.rng.uniform(lower, max_v)
        if operator.value == ">":
            upper = min(max_v, threshold)
            return min_v if upper <= min_v else self.rng.uniform(min_v, upper)
        if operator.value == ">=":
            upper = min(max_v, threshold - 1e-6)
            return min_v if upper <= min_v else self.rng.uniform(min_v, upper)

        return current_value

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply DEFINE constraints and return a modified dataframe."""
        if not self.config.define_spec:
            return df

        define_spec = self.config.define_spec
        df_copy = df.copy()

        for condition in define_spec.independent_conditions:
            self._enforce_independent_condition_exact(df_copy, condition)

        for condition in define_spec.pairwise_conditions:
            self._enforce_pairwise_condition(df_copy, condition)

        # Reconcile independent conditions once more so pairwise edits cannot drift
        # the requested per-row selectivities.
        for condition in define_spec.independent_conditions:
            self._enforce_independent_condition_exact(df_copy, condition)

        return df_copy
