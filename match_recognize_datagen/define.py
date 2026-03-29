"""
DEFINE constraint application logic.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .config import AttributeType, DistributionType, GeneratorConfig


class DefineConstraintApplier:
    """Apply DEFINE independent and pairwise conditions to a dataframe."""

    def __init__(self, config: GeneratorConfig, rng: random.Random):
        self.config = config
        self.rng = rng
        self.attribute_by_name = {attr.name: attr for attr in config.attributes}
        
        # Build distributions and pre-generated values for independent conditions
        self._pre_generated_independent_values: Dict[str, List[Any]] = {}
        self._pre_generated_independent_attrs: Set[str] = set()
        self._independent_distributions: Dict[str, Any] = {}
        self._build_all_independent_distributions()

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _is_supported_inequality(operator_value: str) -> bool:
        return operator_value in {"<", "<=", ">", ">=", "=", "<>"}

    @staticmethod
    def _largest_remainder_counts(probabilities: List[float], total: int) -> List[int]:
        """Allocate integer counts with exact sum(total) using largest remainder."""
        if total <= 0 or not probabilities:
            return [0] * len(probabilities)

        raw = [max(0.0, p) * total for p in probabilities]
        base = [int(x) for x in raw]
        remainder = total - sum(base)

        if remainder > 0:
            order = sorted(
                range(len(raw)),
                key=lambda i: (raw[i] - base[i]),
                reverse=True,
            )
            for i in order[:remainder]:
                base[i] += 1

        return base

    def _build_partition_distribution(
        self, attr, conditions: List[Any]
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Build interval probability masses for one numerical attribute.

        It fits a monotone CDF over condition thresholds so generated values satisfy
        independent inequality selectivities as closely as possible.
        """
        if attr.min_value is None or attr.max_value is None:
            return None

        domain_min = float(attr.min_value)
        domain_max = float(attr.max_value)
        if domain_max <= domain_min:
            return None

        by_x: Dict[float, List[float]] = {
            domain_min: [0.0],
            domain_max: [1.0],
        }
        point_mass_targets: Dict[float, List[float]] = {}

        for cond in conditions:
            op = cond.operator.value
            if not self._is_supported_inequality(op):
                return None

            try:
                threshold = float(cond.value)
            except (TypeError, ValueError):
                return None

            threshold = min(domain_max, max(domain_min, threshold))
            sel = self._clamp01(cond.selectivity)

            if op in {"<", "<="}:
                by_x.setdefault(threshold, []).append(sel)
            elif op in {">", ">="}:
                by_x.setdefault(threshold, []).append(1.0 - sel)
            elif op == "=":
                point_mass_targets.setdefault(threshold, []).append(sel)
                by_x.setdefault(threshold, [])
            else:  # op == "<>"
                point_mass_targets.setdefault(threshold, []).append(1.0 - sel)
                by_x.setdefault(threshold, [])

        xs = sorted(by_x.keys())
        cdf: List[float] = []
        for idx, x in enumerate(xs):
            vals = by_x[x]
            if vals:
                cdf.append(self._clamp01(sum(vals) / len(vals)))
            if x in point_mass_targets:
                vals = point_mass_targets[x]
                p_point = self._clamp01(sum(vals) / len(vals))
                if not by_x[x]:
                    # adjusting the p of interval up to point mass to be more uniform compared with the next interval, if not given [last threshold]<--p-->[point mass]
                    vals_after = by_x[xs[idx+1]]
                    p_after = self._clamp01(sum(vals_after) / len(vals_after))
                    #scaling = (prob danach - prob point - prob davor) * abs(x_point- x_davor)/abs(x_danach-x_davor)
                    scaling = (p_after - p_point - cdf[-1]) * abs((x - xs[idx-1])/(xs[idx+1]-xs[idx-1]))
                    cdf.append(cdf[-1] + p_point + scaling)
                else:
                    cdf[-1] += self._clamp01(sum(vals) / len(vals))

        cdf[0] = 0.0
        cdf[-1] = 1.0
        for i in range(1, len(cdf)):
            cdf[i] = max(cdf[i], cdf[i - 1])
        cdf[-1] = 1.0

        point_p_by_x = {
            x: self._clamp01(sum(targets) / len(targets))
            for x, targets in point_mass_targets.items()
            if targets
        }

        interval_entries: List[Tuple[float, float, float]] = []
        for i in range(len(xs) - 1):
            left = xs[i]
            right = xs[i + 1]
            if right < left:
                continue
            p = max(0.0, cdf[i + 1] - cdf[i])
            point_p = point_p_by_x.get(right, 0.0)
            if point_p > 0.0:
                point_part = min(point_p, p)
                interval_part = max(0.0, p - point_part)
                if interval_part > 0.0:
                    interval_entries.append((left, right, interval_part))
                if point_part > 0.0:
                    interval_entries.append((right, right, point_part))
            else:
                interval_entries.append((left, right, p))

        total_p = sum(p for _, _, p in interval_entries)
        if total_p <= 0.0:
            return None
        return [
            (left, right, p / total_p)
            for left, right, p in interval_entries
            if p > 0.0
        ]

    def _build_categorical_distribution(
        self, attr, conditions: List[Any]
    ) -> Optional[List[Tuple[str, float]]]:
        """
        Build category probability masses for one categorical attribute.

        It computes selectivity targets for each category based on conditions
        and distributes probability accordingly.
        """
        categories = attr.categories or []
        if not categories:
            return None

        # Map each category to its aggregate selectivity target
        category_targets: Dict[str, List[float]] = {}
        for cat in categories:
            category_targets[cat] = []

        for cond in conditions:
            op = cond.operator.value
            target_value = str(cond.value)
            sel = self._clamp01(cond.selectivity)

            # For each category, determine if it satisfies or violates the condition
            for cat in categories:
                if op == "=":
                    satisfies = cat == target_value
                elif op == "<>":
                    satisfies = cat != target_value
                elif op == "<":
                    satisfies = cat < target_value
                elif op == "<=":
                    satisfies = cat <= target_value
                elif op == ">":
                    satisfies = cat > target_value
                elif op == ">=":
                    satisfies = cat >= target_value
                else:
                    return None

                cdf_target = sel if satisfies else (1.0 - sel)
                category_targets[cat].append(cdf_target)

        # Average targets per category
        category_probs = {}
        for cat in categories:
            if category_targets[cat]:
                category_probs[cat] = self._clamp01(
                    sum(category_targets[cat]) / len(category_targets[cat])
                )
            else:
                category_probs[cat] = 1.0 / len(categories)

        # Normalize probabilities
        total_p = sum(category_probs.values())
        if total_p <= 0:
            return None

        category_probs = {cat: p / total_p for cat, p in category_probs.items()}
        return [(cat, p) for cat, p in category_probs.items()]

    @staticmethod
    def _deterministic_values_in_interval(
        left: float, right: float, count: int
    ) -> List[float]:
        """Create deterministic values inside an interval."""
        if count <= 0:
            return []
        if right <= left:
            return [left] * count
        step = (right - left) / count
        return [left + (i + 0.5) * step for i in range(count)]

    def _build_deterministic_values_for_numerical_distribution(
        self,
        distribution: List[Tuple[float, float, float]],
        total_rows: int,
    ) -> List[float]:
        """Generate exact-count values from interval probabilities and shuffle them."""
        probs = [p for _, _, p in distribution]
        counts = self._largest_remainder_counts(probs, total_rows)

        values: List[float] = []
        for (left, right, _), count in zip(distribution, counts):
            values.extend(self._deterministic_values_in_interval(left, right, count))

        self.rng.shuffle(values)
        return values

    def _build_deterministic_values_for_categorical_distribution(
        self,
        distribution: List[Tuple[str, float]],
        total_rows: int,
    ) -> List[str]:
        """Generate exact-count categorical values from probabilities and shuffle them."""
        cats = [cat for cat, _ in distribution]
        probs = [p for _, p in distribution]
        counts = self._largest_remainder_counts(probs, total_rows)

        values: List[str] = []
        for cat, count in zip(cats, counts):
            values.extend([cat] * count)

        self.rng.shuffle(values)
        return values

    def _build_all_independent_distributions(self) -> None:
        """Build distributions and pre-generate values for all supported independent conditions."""
        if not self.config.define_spec:
            return

        # Group conditions by attribute
        by_attr: Dict[str, List[Any]] = {}
        for cond in self.config.define_spec.independent_conditions:
            by_attr.setdefault(cond.attribute_name, []).append(cond)

        for attr_name, conds in by_attr.items():
            attr_cfg = next(
                (a for a in self.config.attributes if a.name == attr_name), None
            )
            if not attr_cfg:
                continue

            if attr_cfg.attr_type == AttributeType.NUMERICAL:
                distribution = self._build_partition_distribution(attr_cfg, conds)
                if distribution is None:
                    continue

                self._independent_distributions[attr_name] = distribution
                self._pre_generated_independent_values[attr_name] = (
                    self._build_deterministic_values_for_numerical_distribution(
                        distribution, self.config.total_rows
                    )
                )
                self._pre_generated_independent_attrs.add(attr_name)

            elif attr_cfg.attr_type == AttributeType.CATEGORICAL:
                distribution = self._build_categorical_distribution(attr_cfg, conds)
                if distribution is None:
                    continue

                self._independent_distributions[attr_name] = distribution
                self._pre_generated_independent_values[attr_name] = (
                    self._build_deterministic_values_for_categorical_distribution(
                        distribution, self.config.total_rows
                    )
                )
                self._pre_generated_independent_attrs.add(attr_name)

    def get_pre_generated_independent_values(self) -> Dict[str, List[Any]]:
        """Return the pre-generated independent values."""
        return self._pre_generated_independent_values

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

        # Independent conditions are already enforced during row generation
        # (values are pre-generated in _build_all_independent_distributions).
        # Only apply pairwise and reconcile after pairwise edits.

        for condition in define_spec.pairwise_conditions:
            self._enforce_pairwise_condition(df_copy, condition)

        # Reconcile independent conditions once more so pairwise edits cannot drift
        # the requested per-row selectivities.
        # for condition in define_spec.independent_conditions:
        #     self._enforce_independent_condition_exact(df_copy, condition)

        return df_copy
