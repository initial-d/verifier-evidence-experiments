#!/usr/bin/env python3
"""Executable fixed-candidate verifier-source experiment.

This experiment isolates the paper's core real-world failure mode: several
validators can agree because they inherited the same incomplete interpretation
of a requirement.  Each row freezes a concrete Python implementation before any
validation decision.  Cheap validators either share the agent's interpretation
or independently read the original requirement.  The online policies then
decide whether to accept, reject, or buy one independent probe before the final
hidden evaluator scores the frozen candidate.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "experiments" / "results"
JSON_OUT = RESULT_DIR / "shared_misunderstanding_real.json"
SUMMARY_TEX_OUT = RESULT_DIR / "shared_misunderstanding_real_table.tex"
SENSITIVITY_TEX_OUT = RESULT_DIR / "shared_misunderstanding_sensitivity_table.tex"
TRACE_TEX_OUT = RESULT_DIR / "shared_misunderstanding_trace_table.tex"

SEED = 20260917
REPETITIONS = 400
CALIBRATION_N = 48
DELTA = 0.05
DECLINE_LOSS = 0.16
PROBE_COST = 0.035
Z_ONE_SIDED_95 = 1.6448536269514722
POINT_THRESHOLD_GRID = tuple(index / 1000.0 for index in range(0, int(DELTA * 1000) + 1))
ATOMS = (
    "correct",
    "mutation",
    "edge",
    "both",
)
ATOM_SPACE = (
    ("correct", True, True),
    ("mutation", False, True),
    ("edge", True, False),
    ("both", False, False),
)


@dataclass(frozen=True)
class Candidate:
    task_id: str
    template: str
    interpretation: str
    bug_family: str
    source: str
    visible_input: Any
    probe_inputs: tuple[Any, ...]
    hidden_inputs: tuple[Any, ...]
    reference: Callable[[Any], Any]


def _sorted_reference(data: list[int]) -> list[int]:
    return sorted(data)


def _checksum_reference(data: list[int]) -> list[int]:
    return list(data) + [sum(data)]


def _defaults_reference(data: dict[str, Any]) -> dict[str, Any]:
    return {"timeout": 30, "retries": 3, **data}


def _active_names_reference(data: list[dict[str, Any]]) -> list[str]:
    return [row["name"].strip().lower() for row in data if row.get("active", False)]


def _tag_reference(data: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in data:
        tag = item.strip().lower()
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _stock_reference(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"sku": row["sku"], "available": row["stock"] - row.get("reserved", 0)}
        for row in data
        if row["stock"] - row.get("reserved", 0) > 0
    ]


TEMPLATES: dict[str, dict[str, Any]] = {
    "sort_numbers": {
        "reference": _sorted_reference,
        "visible": [3, 1, 2],
        "probe": ([2, 2, 1],),
        "hidden": ([3, 1, 2, 1], [], [-2, 5, -2]),
        "correct": "def solve(data):\n    return sorted(data)\n",
        "mutation": "def solve(data):\n    data.sort()\n    return data\n",
        "edge": "def solve(data):\n    return sorted(set(data))\n",
        "both": "def solve(data):\n    data.sort()\n    return data[:-1]\n",
    },
    "append_checksum": {
        "reference": _checksum_reference,
        "visible": [4, 5],
        "probe": ([2, 2],),
        "hidden": ([4, 5, 4], [], [-1, 6]),
        "correct": "def solve(data):\n    return list(data) + [sum(data)]\n",
        "mutation": "def solve(data):\n    data.append(sum(data))\n    return data\n",
        "edge": "def solve(data):\n    return list(data) + [sum(set(data))]\n",
        "both": "def solve(data):\n    data.append(sum(set(data)))\n    return data\n",
    },
    "merge_defaults": {
        "reference": _defaults_reference,
        "visible": {"timeout": 10},
        "probe": ({"timeout": 0},),
        "hidden": ({"retries": 0}, {}, {"timeout": 5, "retries": 0}),
        "correct": "def solve(data):\n    return {'timeout': 30, 'retries': 3, **data}\n",
        "mutation": "def solve(data):\n    data.setdefault('timeout', 30)\n    data.setdefault('retries', 3)\n    return data\n",
        "edge": (
            "def solve(data):\n"
            "    out = {'timeout': 30, 'retries': 3}\n"
            "    for key, value in data.items():\n"
            "        if value:\n"
            "            out[key] = value\n"
            "    return out\n"
        ),
        "both": (
            "def solve(data):\n"
            "    data.setdefault('timeout', 30)\n"
            "    data.setdefault('retries', 3)\n"
            "    return {key: value for key, value in data.items() if value}\n"
        ),
    },
    "active_names": {
        "reference": _active_names_reference,
        "visible": [{"name": " Ana ", "active": True}, {"name": "Bo", "active": True}],
        "probe": ([{"name": " Ana ", "active": True}, {"name": "Cid", "active": False}],),
        "hidden": (
            [{"name": " DEE ", "active": True}],
            [{"name": "Eve", "active": False}, {"name": "Fox", "active": True}],
        ),
        "correct": (
            "def solve(data):\n"
            "    return [row['name'].strip().lower() for row in data if row.get('active', False)]\n"
        ),
        "mutation": (
            "def solve(data):\n"
            "    out = []\n"
            "    for row in data:\n"
            "        row['name'] = row['name'].strip().lower()\n"
            "        if row.get('active', False):\n"
            "            out.append(row['name'])\n"
            "    return out\n"
        ),
        "edge": "def solve(data):\n    return [row['name'].strip().lower() for row in data]\n",
        "both": (
            "def solve(data):\n"
            "    for row in data:\n"
            "        row['name'] = row['name'].strip().lower()\n"
            "    return [row['name'] for row in data]\n"
        ),
    },
    "normalize_tags": {
        "reference": _tag_reference,
        "visible": [" Red ", "Blue"],
        "probe": (["Red", "red"],),
        "hidden": ([" A ", "B", "a"], ["", " C ", "c"]),
        "correct": (
            "def solve(data):\n"
            "    seen = set()\n"
            "    out = []\n"
            "    for item in data:\n"
            "        tag = item.strip().lower()\n"
            "        if tag and tag not in seen:\n"
            "            seen.add(tag)\n"
            "            out.append(tag)\n"
            "    return out\n"
        ),
        "mutation": (
            "def solve(data):\n"
            "    for index, item in enumerate(data):\n"
            "        data[index] = item.strip().lower()\n"
            "    seen = set()\n"
            "    out = []\n"
            "    for tag in data:\n"
            "        if tag and tag not in seen:\n"
            "            seen.add(tag)\n"
            "            out.append(tag)\n"
            "    return out\n"
        ),
        "edge": "def solve(data):\n    return [item.strip().lower() for item in data if item.strip()]\n",
        "both": (
            "def solve(data):\n"
            "    for index, item in enumerate(data):\n"
            "        data[index] = item.strip().lower()\n"
            "    return [item for item in data if item]\n"
        ),
    },
    "available_stock": {
        "reference": _stock_reference,
        "visible": [{"sku": "a", "stock": 3, "reserved": 1}],
        "probe": ([{"sku": "a", "stock": 1, "reserved": 1}, {"sku": "b", "stock": 5}],),
        "hidden": (
            [{"sku": "c", "stock": 2, "reserved": 5}],
            [{"sku": "d", "stock": 0}, {"sku": "e", "stock": 2}],
        ),
        "correct": (
            "def solve(data):\n"
            "    return [\n"
            "        {'sku': row['sku'], 'available': row['stock'] - row.get('reserved', 0)}\n"
            "        for row in data\n"
            "        if row['stock'] - row.get('reserved', 0) > 0\n"
            "    ]\n"
        ),
        "mutation": (
            "def solve(data):\n"
            "    out = []\n"
            "    for row in data:\n"
            "        row['available'] = row['stock'] - row.get('reserved', 0)\n"
            "        if row['available'] > 0:\n"
            "            out.append({'sku': row['sku'], 'available': row['available']})\n"
            "    return out\n"
        ),
        "edge": (
            "def solve(data):\n"
            "    return [\n"
            "        {'sku': row['sku'], 'available': row['stock'] - row.get('reserved', 0)}\n"
            "        for row in data\n"
            "    ]\n"
        ),
        "both": (
            "def solve(data):\n"
            "    for row in data:\n"
            "        row['available'] = row['stock'] - row.get('reserved', 0)\n"
            "    return [{'sku': row['sku'], 'available': row['available']} for row in data]\n"
        ),
    },
}


def _load_solve(source: str) -> Callable[[Any], Any]:
    namespace: dict[str, Any] = {}
    exec(source, namespace)
    return namespace["solve"]


def _run_case(candidate: Candidate, input_value: Any) -> tuple[bool, bool]:
    before = copy.deepcopy(input_value)
    argument = copy.deepcopy(input_value)
    try:
        result = _load_solve(candidate.source)(argument)
    except Exception:
        return False, argument != before
    expected = candidate.reference(copy.deepcopy(input_value))
    return result == expected, argument != before


def _inputs_for_scope(candidate: Candidate, scope: str) -> tuple[Any, ...]:
    if scope == "visible":
        return (candidate.visible_input,)
    if scope == "probe":
        return candidate.probe_inputs
    if scope == "final":
        return candidate.hidden_inputs
    raise ValueError(scope)


def _return_passes(candidate: Candidate, *, scope: str) -> bool:
    inputs = _inputs_for_scope(candidate, scope)
    return all(_run_case(candidate, value)[0] for value in inputs)


def _mutation_passes(candidate: Candidate, *, scope: str) -> bool:
    inputs = _inputs_for_scope(candidate, scope)
    return all(not _run_case(candidate, value)[1] for value in inputs)


def _final_correct(candidate: Candidate) -> bool:
    return _return_passes(candidate, scope="final") and _mutation_passes(candidate, scope="final")


def _cheap_outcomes(candidate: Candidate, condition: str) -> tuple[bool, bool]:
    if condition == "shared":
        if candidate.interpretation == "return_only":
            generated_test = _return_passes(candidate, scope="visible")
            review = candidate.bug_family in {"correct", "mutation"}
        elif candidate.interpretation == "simple_case":
            generated_test = _return_passes(candidate, scope="visible")
            review = candidate.bug_family in {"correct", "edge"}
        else:
            generated_test = _return_passes(candidate, scope="visible") and _mutation_passes(candidate, scope="visible")
            review = candidate.bug_family == "correct"
        return generated_test, review
    if condition == "independent":
        generated_test = _return_passes(candidate, scope="visible") and _mutation_passes(candidate, scope="visible")
        if candidate.bug_family in {"mutation", "both"}:
            review = False
        elif candidate.bug_family == "edge":
            review = candidate.template in {"sort_numbers", "merge_defaults", "normalize_tags"}
        else:
            review = True
        return generated_test, review
    raise ValueError(condition)


def _state(candidate: Candidate) -> str:
    if _final_correct(candidate):
        return "correct"
    mutation_fails = not _mutation_passes(candidate, scope="final")
    return_fails = not _return_passes(candidate, scope="final")
    if mutation_fails and return_fails:
        return "both"
    if mutation_fails:
        return "mutation"
    return "edge"


def _make_candidates() -> list[Candidate]:
    rows: list[Candidate] = []
    bug_schedule = (
        [("correct", "complete")] * 18
        + [("correct", "return_only")] * 18
        + [("correct", "simple_case")] * 18
        + [("mutation", "return_only")] * 2
        + [("edge", "simple_case")] * 2
        + [("both", "return_only")] * 2
    )
    index = 0
    for template_name, template in TEMPLATES.items():
        for repeat, (bug_family, interpretation) in enumerate(bug_schedule):
            rows.append(
                Candidate(
                    task_id=f"shared_real_{index:04d}",
                    template=template_name,
                    interpretation=interpretation,
                    bug_family=bug_family,
                    source=template[bug_family],
                    visible_input=template["visible"],
                    probe_inputs=tuple(template["probe"]),
                    hidden_inputs=tuple(template["hidden"]),
                    reference=template["reference"],
                )
            )
            index += 1
    return rows


def _wilson_interval(successes: int, n: int, z: float = Z_ONE_SIDED_95) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - radius) / denom), min(1.0, (centre + radius) / denom)


def _beta_integer_cdf(x: float, alpha: int, beta: int) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    total = alpha + beta - 1
    return sum(
        math.comb(total, index) * (x ** index) * ((1.0 - x) ** (total - index))
        for index in range(alpha, total + 1)
    )


def _beta_posterior_upper(successes: int, n: int, quantile: float = 0.95) -> float:
    if n <= 0:
        return 1.0
    alpha = successes + 1
    beta = n - successes + 1
    low = 0.0
    high = 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if _beta_integer_cdf(mid, alpha, beta) < quantile:
            low = mid
        else:
            high = mid
    return high


def _record(candidate: Candidate, condition: str) -> dict[str, Any]:
    cheap_test, cheap_review = _cheap_outcomes(candidate, condition)
    return {
        "task_id": candidate.task_id,
        "template": candidate.template,
        "condition": condition,
        "interpretation": candidate.interpretation,
        "bug_family": candidate.bug_family,
        "state": _state(candidate),
        "cheap_test_pass": cheap_test,
        "cheap_review_pass": cheap_review,
        "cheap_all_pass": cheap_test and cheap_review,
        "mutation_probe_pass": _mutation_passes(candidate, scope="probe"),
        "edge_probe_pass": _return_passes(candidate, scope="probe"),
        "final_correct": _final_correct(candidate),
    }


def _atom_space_for_history(history: tuple[str, str]) -> tuple[tuple[str, bool, bool], ...]:
    condition, interpretation = history
    if condition == "shared" and interpretation == "return_only":
        return (("correct", True, True), ("mutation", False, True))
    if condition == "shared" and interpretation == "simple_case":
        return (("correct", True, True), ("edge", True, False))
    if condition == "shared" and interpretation == "complete":
        return (("correct", True, True),)
    if condition == "independent":
        return (("correct", True, True), ("edge", True, False))
    return ATOM_SPACE


def _fit_point(records: list[dict[str, Any]], history: tuple[str, str]) -> dict[tuple[str, bool, bool], float]:
    matching = [
        row for row in records
        if row["condition"] == history[0]
        and row["interpretation"] == history[1]
        and row["cheap_all_pass"]
    ]
    atom_space = _atom_space_for_history(history)
    atom_count = len(atom_space)
    denominator = len(matching) + 0.5 * atom_count
    distribution: dict[tuple[str, bool, bool], float] = {}
    for state, mutation_pass, edge_pass in atom_space:
        count = sum(
            row["state"] == state
            and row["mutation_probe_pass"] == mutation_pass
            and row["edge_probe_pass"] == edge_pass
            for row in matching
        )
        distribution[(state, mutation_pass, edge_pass)] = (count + 0.5) / denominator
    return distribution


def _fit_intervals(records: list[dict[str, Any]], history: tuple[str, str]) -> dict[tuple[str, bool, bool], tuple[float, float]]:
    matching = [
        row for row in records
        if row["condition"] == history[0]
        and row["interpretation"] == history[1]
        and row["cheap_all_pass"]
    ]
    intervals: dict[tuple[str, bool, bool], tuple[float, float]] = {}
    for state, mutation_pass, edge_pass in _atom_space_for_history(history):
        count = sum(
            row["state"] == state
            and row["mutation_probe_pass"] == mutation_pass
            and row["edge_probe_pass"] == edge_pass
            for row in matching
        )
        intervals[(state, mutation_pass, edge_pass)] = _wilson_interval(count, len(matching))
    lower_sum = sum(low for low, _ in intervals.values())
    upper_sum = sum(high for _, high in intervals.values())
    if lower_sum > 1.0 or upper_sum < 1.0:
        return {atom: (0.0, 1.0) for atom in intervals}
    return intervals


def _wrong_upper(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 1.0
    wrong = sum(row["state"] != "correct" for row in rows)
    return _beta_posterior_upper(wrong, len(rows))


def _fit_upper(records: list[dict[str, Any]], history: tuple[str, str]) -> dict[str, Any]:
    matching = [
        row for row in records
        if row["condition"] == history[0]
        and row["interpretation"] == history[1]
        and row["cheap_all_pass"]
    ]
    post: dict[tuple[str, bool], float] = {}
    for query in ("mutation_probe", "edge_probe"):
        key = query + "_pass"
        for passed in (True, False):
            post[(query, passed)] = _wrong_upper([
                row for row in matching if row[key] == passed
            ])
    return {
        "pre": _wrong_upper(matching),
        "post": post,
        "n": len(matching),
    }


def _risk_point(distribution: dict[tuple[str, bool, bool], float]) -> float:
    return sum(prob for atom, prob in distribution.items() if atom[0] != "correct")


def _maximize_linear(
    intervals: dict[tuple[str, bool, bool], tuple[float, float]],
    coefficient: Callable[[tuple[str, bool, bool]], float],
) -> float:
    allocation = {atom: low for atom, (low, _) in intervals.items()}
    remaining = 1.0 - sum(allocation.values())
    ordered = sorted(intervals, key=coefficient, reverse=True)
    for atom in ordered:
        if remaining <= 1e-12:
            break
        low, high = intervals[atom]
        extra = min(high - low, remaining)
        allocation[atom] += extra
        remaining -= extra
    return sum(allocation[atom] * coefficient(atom) for atom in intervals)


def _risk_upper(intervals: dict[tuple[str, bool, bool], tuple[float, float]]) -> float:
    return _maximize_linear(intervals, lambda atom: 1.0 if atom[0] != "correct" else 0.0)


def _conditional_risk_upper(
    intervals: dict[tuple[str, bool, bool], tuple[float, float]],
    query: str,
    passed: bool,
) -> float:
    relevant_index = 1 if query == "mutation_probe" else 2
    numerator_intervals = {
        atom: bounds
        for atom, bounds in intervals.items()
        if atom[relevant_index] == passed
    }
    lower_denominator = sum(low for low, _ in numerator_intervals.values())
    upper_denominator = sum(high for _, high in numerator_intervals.values())
    outside_lower = sum(
        low for atom, (low, _) in intervals.items()
        if atom[relevant_index] != passed
    )
    max_feasible_denominator = min(upper_denominator, 1.0 - outside_lower)
    if max_feasible_denominator <= 1e-12:
        return 1.0
    wrong_upper = _maximize_linear(
        numerator_intervals,
        lambda atom: 1.0 if atom[0] != "correct" else 0.0,
    )
    return min(1.0, wrong_upper / max_feasible_denominator)


def _point_after_query(
    distribution: dict[tuple[str, bool, bool], float],
    query: str,
    passed: bool,
) -> float:
    relevant_index = 1 if query == "mutation_probe" else 2
    denominator = sum(prob for atom, prob in distribution.items() if atom[relevant_index] == passed)
    if denominator <= 0.0:
        return 1.0
    numerator = sum(
        prob
        for atom, prob in distribution.items()
        if atom[0] != "correct" and atom[relevant_index] == passed
    )
    return numerator / denominator


def _point_query_loss(
    distribution: dict[tuple[str, bool, bool], float],
    query: str,
    delta: float = DELTA,
) -> float:
    relevant_index = 1 if query == "mutation_probe" else 2
    pass_probability = sum(prob for atom, prob in distribution.items() if atom[relevant_index])
    pass_risk = _point_after_query(distribution, query, True)
    if pass_risk <= delta:
        return PROBE_COST + pass_probability * pass_risk + (1.0 - pass_probability) * DECLINE_LOSS
    return PROBE_COST + DECLINE_LOSS


def _upper_query_loss(
    distribution: dict[tuple[str, bool, bool], float],
    upper: dict[str, Any],
    query: str,
    delta: float = DELTA,
) -> float:
    relevant_index = 1 if query == "mutation_probe" else 2
    pass_probability = sum(prob for atom, prob in distribution.items() if atom[relevant_index])
    post_upper = upper["post"][(query, True)]
    if post_upper <= delta:
        pass_risk = _point_after_query(distribution, query, True)
        return PROBE_COST + pass_probability * pass_risk + (1.0 - pass_probability) * DECLINE_LOSS
    return PROBE_COST + DECLINE_LOSS


def _credal_query_loss(
    intervals: dict[tuple[str, bool, bool], tuple[float, float]],
    query: str,
    delta: float = DELTA,
) -> float:
    pass_risk = _conditional_risk_upper(intervals, query, True)
    if pass_risk > delta:
        return PROBE_COST + DECLINE_LOSS
    relevant_index = 1 if query == "mutation_probe" else 2
    return PROBE_COST + _maximize_linear(
        intervals,
        lambda atom: (
            1.0
            if atom[relevant_index] and atom[0] != "correct"
            else DECLINE_LOSS
            if not atom[relevant_index]
            else 0.0
        ),
    )


def _select_point(
    distribution: dict[tuple[str, bool, bool], float],
    delta: float = DELTA,
) -> tuple[str, float]:
    current_risk = _risk_point(distribution)
    if current_risk <= delta:
        return "accept", current_risk
    losses = {
        "reject": DECLINE_LOSS,
        "mutation_probe": _point_query_loss(distribution, "mutation_probe", delta),
        "edge_probe": _point_query_loss(distribution, "edge_probe", delta),
    }
    action, loss = min(losses.items(), key=lambda item: item[1])
    return (action if loss < DECLINE_LOSS else "reject"), current_risk


def _select_upper(
    distribution: dict[tuple[str, bool, bool], float],
    upper: dict[str, Any],
    delta: float = DELTA,
) -> tuple[str, float]:
    current_risk = float(upper["pre"])
    if current_risk <= delta:
        return "accept", current_risk
    losses = {
        "reject": DECLINE_LOSS,
        "mutation_probe": _upper_query_loss(distribution, upper, "mutation_probe", delta),
        "edge_probe": _upper_query_loss(distribution, upper, "edge_probe", delta),
    }
    action, loss = min(losses.items(), key=lambda item: item[1])
    return (action if loss < DECLINE_LOSS else "reject"), current_risk


def _select_credal(
    intervals: dict[tuple[str, bool, bool], tuple[float, float]],
    delta: float = DELTA,
) -> tuple[str, float]:
    current_risk = _risk_upper(intervals)
    if current_risk <= delta:
        return "accept", current_risk
    losses = {
        "reject": DECLINE_LOSS,
        "mutation_probe": _credal_query_loss(intervals, "mutation_probe", delta),
        "edge_probe": _credal_query_loss(intervals, "edge_probe", delta),
    }
    action, loss = min(losses.items(), key=lambda item: item[1])
    return (action if loss < DECLINE_LOSS else "reject"), current_risk


def _apply_policy(
    *,
    policy: str,
    row: dict[str, Any],
    point: dict[tuple[str, bool, bool], float],
    intervals: dict[tuple[str, bool, bool], tuple[float, float]],
    upper: dict[str, Any] | None = None,
    bayes_delta: float = DELTA,
) -> dict[str, Any]:
    if not row["cheap_all_pass"]:
        return {
            "accepted": False,
            "cost": 0.0,
            "query": "none",
            "query_count": 0,
            "risk": 1.0,
            "post_risk": None,
        }
    if policy in {"bayesian_point", "bayesian_cost_matched"}:
        action, risk = _select_point(point, bayes_delta)
        if action == "accept":
            return {
                "accepted": True,
                "cost": 0.0,
                "query": "none",
                "query_count": 0,
                "risk": risk,
                "post_risk": risk,
            }
        if action == "reject":
            return {
                "accepted": False,
                "cost": 0.0,
                "query": "none",
                "query_count": 0,
                "risk": risk,
                "post_risk": None,
            }
        passed = bool(row[action + "_pass"])
        post_risk = _point_after_query(point, action, passed)
        return {
            "accepted": passed and post_risk <= bayes_delta,
            "cost": PROBE_COST,
            "query": action,
            "query_count": 1,
            "risk": risk,
            "post_risk": post_risk,
        }
    if policy == "bayesian_upper":
        if upper is None:
            raise ValueError("bayesian_upper requires fitted upper-bound statistics")
        action, risk = _select_upper(point, upper)
        if action == "accept":
            return {
                "accepted": True,
                "cost": 0.0,
                "query": "none",
                "query_count": 0,
                "risk": risk,
                "post_risk": risk,
            }
        if action == "reject":
            return {
                "accepted": False,
                "cost": 0.0,
                "query": "none",
                "query_count": 0,
                "risk": risk,
                "post_risk": None,
            }
        passed = bool(row[action + "_pass"])
        post_risk = upper["post"][(action, passed)]
        return {
            "accepted": passed and post_risk <= DELTA,
            "cost": PROBE_COST,
            "query": action,
            "query_count": 1,
            "risk": risk,
            "post_risk": post_risk,
        }
    if policy == "dynamic_credal":
        action, risk = _select_credal(intervals)
        if action == "accept":
            return {
                "accepted": True,
                "cost": 0.0,
                "query": "none",
                "query_count": 0,
                "risk": risk,
                "post_risk": risk,
            }
        if action == "reject":
            return {
                "accepted": False,
                "cost": 0.0,
                "query": "none",
                "query_count": 0,
                "risk": risk,
                "post_risk": None,
            }
        passed = bool(row[action + "_pass"])
        post_risk = _conditional_risk_upper(intervals, action, passed)
        return {
            "accepted": passed and post_risk <= DELTA,
            "cost": PROBE_COST,
            "query": action,
            "query_count": 1,
            "risk": risk,
            "post_risk": post_risk,
        }
    if policy == "fixed_independent_probe":
        passed = bool(row["mutation_probe_pass"])
        return {
            "accepted": passed,
            "cost": PROBE_COST,
            "query": "mutation_probe",
            "query_count": 1,
            "risk": None,
            "post_risk": None,
        }
    if policy == "fixed_two_probe":
        mutation_passed = bool(row["mutation_probe_pass"])
        if not mutation_passed:
            return {
                "accepted": False,
                "cost": PROBE_COST,
                "query": "mutation_probe",
                "query_count": 1,
                "risk": None,
                "post_risk": None,
            }
        edge_passed = bool(row["edge_probe_pass"])
        return {
            "accepted": edge_passed,
            "cost": 2.0 * PROBE_COST,
            "query": "mutation_probe+edge_probe",
            "query_count": 2,
            "risk": None,
            "post_risk": None,
        }
    raise ValueError(policy)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    accepted = [row for row in rows if row["accepted"]]
    wrong = [row for row in accepted if not row["final_correct"]]
    correct = [row for row in accepted if row["final_correct"]]
    return {
        "tasks": len(rows),
        "accept_rate": mean(float(row["accepted"]) for row in rows),
        "error_among_accepted": len(wrong) / len(accepted) if accepted else None,
        "wrong_accept_per_task": len(wrong) / len(rows),
        "correct_completion_rate": len(correct) / len(rows),
        "mean_cost": mean(float(row["cost"]) for row in rows),
        "query_rate": mean(float(row["query"] != "none") for row in rows),
        "mean_probes": mean(float(row.get("query_count", float(row["query"] != "none"))) for row in rows),
        "mutation_query_rate": mean(float(row["query"] == "mutation_probe") for row in rows),
        "edge_query_rate": mean(float(row["query"] == "edge_probe") for row in rows),
    }


def _paired_summary(records: list[dict[str, Any]], condition: str) -> dict[str, float | int]:
    by_key: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for row in records:
        if row["condition"] != condition:
            continue
        key = (row["rep"], row["task_id"])
        by_key.setdefault(key, {})[row["policy"]] = row
    pairs = [rows for rows in by_key.values() if {"bayesian_point", "dynamic_credal"} <= set(rows)]
    different = [rows for rows in pairs if rows["bayesian_point"]["accepted"] != rows["dynamic_credal"]["accepted"]]
    avoided_wrong = sum(
        rows["bayesian_point"]["accepted"]
        and not rows["dynamic_credal"]["accepted"]
        and not rows["bayesian_point"]["final_correct"]
        for rows in different
    )
    lost_correct = sum(
        rows["bayesian_point"]["accepted"]
        and not rows["dynamic_credal"]["accepted"]
        and rows["bayesian_point"]["final_correct"]
        for rows in different
    )
    gained_correct = sum(
        (not rows["bayesian_point"]["accepted"])
        and rows["dynamic_credal"]["accepted"]
        and rows["dynamic_credal"]["final_correct"]
        for rows in different
    )
    extra_cost = mean(
        rows["dynamic_credal"]["cost"] - rows["bayesian_point"]["cost"]
        for rows in pairs
    )
    return {
        "paired_tasks": len(pairs),
        "decision_disagreements": len(different),
        "avoided_wrong_accepts": avoided_wrong,
        "lost_correct_accepts": lost_correct,
        "gained_correct_accepts": gained_correct,
        "mean_extra_cost": extra_cost,
    }


def _apply_rows(
    *,
    rows: list[dict[str, Any]],
    policy: str,
    fitted_point: dict[tuple[str, str], dict[tuple[str, bool, bool], float]],
    fitted_intervals: dict[tuple[str, str], dict[tuple[str, bool, bool], tuple[float, float]]],
    fitted_upper: dict[tuple[str, str], dict[str, Any]],
    bayes_delta_by_condition: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    decision_rows: list[dict[str, Any]] = []
    for row in rows:
        history = (row["condition"], row["interpretation"])
        decision = _apply_policy(
            policy=policy,
            row=row,
            point=fitted_point[history],
            intervals=fitted_intervals[history],
            upper=fitted_upper[history],
            bayes_delta=(
                bayes_delta_by_condition[row["condition"]]
                if bayes_delta_by_condition and policy == "bayesian_cost_matched"
                else DELTA
            ),
        )
        decision_rows.append({**row, **decision, "policy": policy})
    return decision_rows


def _select_cost_matched_threshold(
    *,
    condition: str,
    calibration: list[dict[str, Any]],
    fitted_point: dict[tuple[str, str], dict[tuple[str, bool, bool], float]],
    fitted_intervals: dict[tuple[str, str], dict[tuple[str, bool, bool], tuple[float, float]]],
    fitted_upper: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, float]:
    rows = [row for row in calibration if row["condition"] == condition]
    target_summary = _summarize(_apply_rows(
        rows=rows,
        policy="dynamic_credal",
        fitted_point=fitted_point,
        fitted_intervals=fitted_intervals,
        fitted_upper=fitted_upper,
    ))
    target_cost = float(target_summary["mean_cost"])
    target_completion = float(target_summary["correct_completion_rate"])
    best: dict[str, float] | None = None
    for threshold in POINT_THRESHOLD_GRID:
        summary = _summarize(_apply_rows(
            rows=rows,
            policy="bayesian_cost_matched",
            fitted_point=fitted_point,
            fitted_intervals=fitted_intervals,
            fitted_upper=fitted_upper,
            bayes_delta_by_condition={condition: threshold},
        ))
        score = (
            abs(float(summary["mean_cost"]) - target_cost),
            abs(float(summary["correct_completion_rate"]) - target_completion),
            float(summary["wrong_accept_per_task"]),
            -threshold,
        )
        if best is None or score < best["score"]:
            best = {
                "threshold": threshold,
                "target_cost": target_cost,
                "target_completion": target_completion,
                "calibration_cost": float(summary["mean_cost"]),
                "calibration_correct_completion": float(summary["correct_completion_rate"]),
                "calibration_wrong_accept": float(summary["wrong_accept_per_task"]),
                "score": score,
            }
    assert best is not None
    return best


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _bootstrap_mean_ci(values: list[float], rng: random.Random, draws: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(sample))
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _build_rows() -> list[dict[str, Any]]:
    return [
        _record(candidate, condition)
        for candidate in _make_candidates()
        for condition in ("shared", "independent")
    ]


def run() -> dict[str, Any]:
    rng = random.Random(SEED)
    all_rows = _build_rows()
    policies = (
        "bayesian_point",
        "bayesian_cost_matched",
        "bayesian_upper",
        "fixed_independent_probe",
        "fixed_two_probe",
        "dynamic_credal",
    )
    decision_records: list[dict[str, Any]] = []
    selected_thresholds: dict[str, list[dict[str, float]]] = {
        condition: [] for condition in ("shared", "independent")
    }
    repetition_deltas: dict[str, dict[str, list[float]]] = {
        condition: {"wrong_accept": [], "correct_completion": [], "cost": []}
        for condition in ("shared", "independent")
    }
    candidates_by_condition = {
        condition: [row for row in all_rows if row["condition"] == condition]
        for condition in ("shared", "independent")
    }
    for rep in range(REPETITIONS):
        calibration_ids: set[str] = set()
        for condition, rows in candidates_by_condition.items():
            ids = [row["task_id"] for row in rows]
            calibration_ids.update(f"{condition}:{task_id}" for task_id in rng.sample(ids, CALIBRATION_N))
        calibration = [
            row for row in all_rows
            if f"{row['condition']}:{row['task_id']}" in calibration_ids
        ]
        test_rows = [
            row for row in all_rows
            if f"{row['condition']}:{row['task_id']}" not in calibration_ids
        ]
        fitted_point: dict[tuple[str, str], dict[tuple[str, bool, bool], float]] = {}
        fitted_intervals: dict[tuple[str, str], dict[tuple[str, bool, bool], tuple[float, float]]] = {}
        fitted_upper: dict[tuple[str, str], dict[str, Any]] = {}
        for condition in ("shared", "independent"):
            for interpretation in ("return_only", "simple_case", "complete"):
                history = (condition, interpretation)
                fitted_point[history] = _fit_point(calibration, history)
                fitted_intervals[history] = _fit_intervals(calibration, history)
                fitted_upper[history] = _fit_upper(calibration, history)
        bayes_delta_by_condition = {}
        for condition in ("shared", "independent"):
            threshold_record = _select_cost_matched_threshold(
                condition=condition,
                calibration=calibration,
                fitted_point=fitted_point,
                fitted_intervals=fitted_intervals,
                fitted_upper=fitted_upper,
            )
            selected_thresholds[condition].append(threshold_record)
            bayes_delta_by_condition[condition] = threshold_record["threshold"]
        rep_records: list[dict[str, Any]] = []
        for row in test_rows:
            history = (row["condition"], row["interpretation"])
            for policy in policies:
                decision = _apply_policy(
                    policy=policy,
                    row=row,
                    point=fitted_point[history],
                    intervals=fitted_intervals[history],
                    upper=fitted_upper[history],
                    bayes_delta=(
                        bayes_delta_by_condition[row["condition"]]
                        if policy == "bayesian_cost_matched"
                        else DELTA
                    ),
                )
                record = {
                    **row,
                    **decision,
                    "policy": policy,
                    "rep": rep,
                }
                decision_records.append(record)
                rep_records.append(record)
        for condition in ("shared", "independent"):
            bayes = _summarize([
                row for row in rep_records
                if row["condition"] == condition and row["policy"] == "bayesian_point"
            ])
            credal = _summarize([
                row for row in rep_records
                if row["condition"] == condition and row["policy"] == "dynamic_credal"
            ])
            repetition_deltas[condition]["wrong_accept"].append(
                float(bayes["wrong_accept_per_task"]) - float(credal["wrong_accept_per_task"])
            )
            repetition_deltas[condition]["correct_completion"].append(
                float(credal["correct_completion_rate"]) - float(bayes["correct_completion_rate"])
            )
            repetition_deltas[condition]["cost"].append(
                float(credal["mean_cost"]) - float(bayes["mean_cost"])
            )
    summary = {
        condition: {
            policy: _summarize([
                row for row in decision_records
                if row["condition"] == condition and row["policy"] == policy
            ])
            for policy in policies
        }
        for condition in ("shared", "independent")
    }
    diagnostics = {}
    bootstrap_rng = random.Random(SEED + 1)
    for condition, deltas in repetition_deltas.items():
        diagnostics[condition] = {}
        for metric, values in deltas.items():
            low, high = _bootstrap_mean_ci(values, bootstrap_rng)
            diagnostics[condition][metric] = {
                "mean": mean(values),
                "mean_ci_low": low,
                "mean_ci_high": high,
                "central_95_low": _percentile(values, 0.025),
                "central_95_high": _percentile(values, 0.975),
                "improved_repetitions": sum(value > 1e-12 for value in values),
                "worse_repetitions": sum(value < -1e-12 for value in values),
            }
    cheap_failure_rates = {
        condition: {
            "cheap_all_pass_rate": mean(float(row["cheap_all_pass"]) for row in rows),
            "cheap_all_pass_wrong_rate": mean(
                float(row["cheap_all_pass"] and not row["final_correct"]) for row in rows
            ),
            "wrong_given_all_pass": (
                sum(row["cheap_all_pass"] and not row["final_correct"] for row in rows)
                / sum(row["cheap_all_pass"] for row in rows)
            ),
        }
        for condition, rows in candidates_by_condition.items()
    }
    traces = _select_traces(decision_records)
    return {
        "suite": "shared misunderstanding fixed-candidate executable replay",
        "config": {
            "seed": SEED,
            "repetitions": REPETITIONS,
            "calibration_n_per_condition": CALIBRATION_N,
            "test_candidates_per_condition_per_rep": len(_make_candidates()) - CALIBRATION_N,
            "delta": DELTA,
            "point_threshold_grid": [min(POINT_THRESHOLD_GRID), max(POINT_THRESHOLD_GRID)],
            "decline_loss": DECLINE_LOSS,
            "probe_cost": PROBE_COST,
            "online_probe_inputs": "fixed before calibration from the public requirement templates",
            "final_inputs": "held out from the online controller and used only after accept/reject freezes",
        },
        "conditions": {
            "shared": "cheap generated test and review reuse the frozen candidate's incomplete requirement interpretation",
            "independent": "cheap generated test and review are generated from the original requirement",
        },
        "policies": {
            "bayesian_point": "Jeffreys-smoothed plug-in atom distribution from the same calibration histories",
            "bayesian_cost_matched": "history-aware plug-in Bayesian controller with a stricter accept gate selected on calibration to match dynamic-credal probe cost",
            "bayesian_upper": "plug-in query selection with a one-sided calibration upper bound for the accept gate",
            "fixed_independent_probe": "always buys the mutation-preservation probe after cheap passes",
            "fixed_two_probe": "runs the mutation-preservation probe and then the edge-case probe after cheap passes",
            "dynamic_credal": "Wilson interval atom set with the same value rule and risk gate",
        },
        "cheap_failure_rates": cheap_failure_rates,
        "summary": summary,
        "selected_thresholds": {
            condition: {
                "mean_threshold": mean(row["threshold"] for row in rows),
                "min_threshold": min(row["threshold"] for row in rows),
                "max_threshold": max(row["threshold"] for row in rows),
                "mean_calibration_cost": mean(row["calibration_cost"] for row in rows),
                "mean_target_cost": mean(row["target_cost"] for row in rows),
            }
            for condition, rows in selected_thresholds.items()
        },
        "paired_diagnostics": {
            condition: _paired_summary(decision_records, condition)
            for condition in ("shared", "independent")
        },
        "repetition_deltas": diagnostics,
        "traces": traces,
        "records": decision_records,
    }


def _select_traces(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces = []
    selectors = (
        lambda row: row["condition"] == "shared"
        and row["policy"] == "dynamic_credal"
        and row["bug_family"] == "mutation"
        and row["cheap_all_pass"]
        and row["query"] == "mutation_probe"
        and not row["accepted"],
        lambda row: row["condition"] == "shared"
        and row["policy"] == "dynamic_credal"
        and row["bug_family"] == "edge"
        and row["cheap_all_pass"]
        and row["query"] == "edge_probe"
        and not row["accepted"],
        lambda row: row["condition"] == "independent"
        and row["policy"] == "dynamic_credal"
        and row["final_correct"]
        and row["accepted"],
    )
    for selector in selectors:
        for row in records:
            if selector(row) and all(existing["task_id"] != row["task_id"] for existing in traces):
                traces.append(row)
                break
    return traces


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_latex(result: dict[str, Any]) -> None:
    labels = {
        "bayesian_point": "Bayesian point",
        "fixed_independent_probe": "Fixed mutation probe",
        "fixed_two_probe": "Sequential probes",
        "dynamic_credal": "Dynamic credal",
    }
    condition_labels = {
        "shared": "Explanation-conditioned",
        "independent": "Requirement-conditioned",
    }
    lines = [
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Condition & Policy & Accept & Error/accepted & Wrong accept & Correct completion & Cost & Probes\\",
        r"\midrule",
    ]
    for condition in ("shared", "independent"):
        for policy, label in labels.items():
            row = result["summary"][condition][policy]
            lines.append(
                "{} & {} & {} & {} & {} & {} & {} & {}\\\\".format(
                    condition_labels[condition],
                    label,
                    _fmt(row["accept_rate"]),
                    _fmt(row["error_among_accepted"]),
                    _fmt(row["wrong_accept_per_task"]),
                    _fmt(row["correct_completion_rate"]),
                    _fmt(row["mean_cost"]),
                    _fmt(row["mean_probes"]),
                )
            )
        if condition == "shared":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    SUMMARY_TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_sensitivity_latex(result: dict[str, Any]) -> None:
    labels = {
        "dynamic_credal": "Dynamic credal",
        "bayesian_cost_matched": "Bayes cost-matched",
        "bayesian_upper": "Bayesian upper",
    }
    condition_labels = {
        "shared": "Explanation-conditioned",
        "independent": "Requirement-conditioned",
    }
    lines = [
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Condition & Policy & Accept & Error/accepted & Wrong accept & Correct completion & Cost & Probes\\",
        r"\midrule",
    ]
    for condition in ("shared", "independent"):
        for policy, label in labels.items():
            row = result["summary"][condition][policy]
            lines.append(
                "{} & {} & {} & {} & {} & {} & {} & {}\\\\".format(
                    condition_labels[condition],
                    label,
                    _fmt(row["accept_rate"]),
                    _fmt(row["error_among_accepted"]),
                    _fmt(row["wrong_accept_per_task"]),
                    _fmt(row["correct_completion_rate"]),
                    _fmt(row["mean_cost"]),
                    _fmt(row["mean_probes"]),
                )
            )
        threshold = result["selected_thresholds"][condition]
        lines.append(
            r"\multicolumn{8}{@{}l@{}}{"
            + "Bayes cost-matched threshold: mean "
            + _fmt(threshold["mean_threshold"])
            + ", range ["
            + _fmt(threshold["min_threshold"])
            + ", "
            + _fmt(threshold["max_threshold"])
            + r"].}\\"
        )
        if condition == "shared":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    SENSITIVITY_TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_trace_latex(result: dict[str, Any]) -> None:
    lines = [
        r"\begin{tabular}{@{}lllllrl@{}}",
        r"\toprule",
        r"Condition & Template & Candidate & Cheap result & Visible conditioning information & Pre-risk & Route\\",
        r"\midrule",
    ]
    for row in result["traces"]:
        route = "accept" if row["accepted"] else "reject"
        if row["query"] != "none":
            probe_pass = row[row["query"] + "_pass"]
            route = f"{row['query'].replace('_', ' ')} {'pass' if probe_pass else 'fail'}; {route}"
        visible = (
            f"source={row['condition']}; "
            f"interp={str(row['interpretation']).replace('_', '-')}; "
            f"atoms={'/'.join(atom[0] for atom in _atom_space_for_history((row['condition'], row['interpretation'])))}"
        )
        lines.append(
            "{} & {} & {} & pass/pass & {} & {} & {}\\\\".format(
                _tex_escape(str(row["condition"])),
                _tex_escape(str(row["template"]).replace("_", " ")),
                _tex_escape(str(row["bug_family"]).replace("_", " ")),
                _tex_escape(visible),
                _fmt(float(row["risk"])),
                _tex_escape(route),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    TRACE_TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = run()
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_latex(result)
    write_sensitivity_latex(result)
    write_trace_latex(result)
    printable = {key: value for key, value in result.items() if key != "records"}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
