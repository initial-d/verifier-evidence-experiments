#!/usr/bin/env python3
"""Verifier-dependence and active-check pilots.

The script contains three direct mechanism studies for the revised manuscript.

1. A same-distribution finite-sample calibration study: calibration counts for
   common verifier misses are freshly sampled from the same group distribution
   as the test tasks.  The reported coverage is therefore about whether the
   credal upper bound contains the true common-miss probability.

2. A dynamic check-selection attribution study: after repeated cheap passes, the controller
   can buy interface, boundary, broad, or repeated-review evidence.  The value
   rule compares a query against the best currently allowed terminal action,
   and an accept branch is available only when the same risk threshold is met.

3. A coupled dependence-and-active-selection study: finite calibration creates
   intervals over failure-family explanations, and the controller chooses
   checks under those explanation sets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "experiments" / "results"
JSON_OUT = RESULT_DIR / "verifier_dependence_pilot.json"
DEPENDENCE_TEX_OUT = RESULT_DIR / "verifier_dependence_pilot_table.tex"
DYNAMIC_TEX_OUT = RESULT_DIR / "verifier_dynamic_checks_table.tex"
COUPLED_TEX_OUT = RESULT_DIR / "verifier_coupled_active_table.tex"
COUPLED_PATH_TEX_OUT = RESULT_DIR / "verifier_coupled_path_table.tex"
COUPLED_DELTA_TEX_OUT = RESULT_DIR / "verifier_coupled_delta_table.tex"
COUPLED_STABILITY_TEX_OUT = RESULT_DIR / "verifier_coupled_stability_table.tex"
COUPLED_DISAGREEMENT_TEX_OUT = RESULT_DIR / "verifier_coupled_disagreement_table.tex"
SHIFT_TEX_OUT = RESULT_DIR / "verifier_shift_stress_table.tex"
NECESSITY_TEX_OUT = RESULT_DIR / "verifier_necessity_stress_table.tex"
NECESSITY_PATH_TEX_OUT = RESULT_DIR / "verifier_necessity_path_table.tex"
NECESSITY_TRACE_TEX_OUT = RESULT_DIR / "verifier_necessity_trace_table.tex"

SEED = 20260917
CALIBRATION_REPS = 500
TEST_TASKS_PER_REP = 1000
DELTA = 0.05
DECLINE_LOSS = 0.14
CHEAP_CHECK_COST = 0.01
ORTHOGONAL_CHECK_COST = 0.10
TRUE_PASS = 0.80
FALSE_PASS_MARGINAL = 0.20
ORTHOGONAL_TRUE_PASS = 0.90
ORTHOGONAL_FALSE_PASS = 0.05


@dataclass(frozen=True)
class DependenceGroup:
    name: str
    mass: float
    true_common_miss: float
    calibration_n: int


DEPENDENCE_GROUPS = (
    DependenceGroup("independent", 0.50, 0.00, 400),
    DependenceGroup("moderate_shared", 0.30, 0.04, 160),
    DependenceGroup("strong_shared", 0.20, 0.12, 160),
)


DEPENDENCE_POLICIES = (
    "independent_bayes",
    "correlated_plugin",
    "bayesian_upper",
    "credal_dependence",
    "dedup_then_orthogonal",
    "always_orthogonal",
    "abstain_all",
)


@dataclass(frozen=True)
class Query:
    name: str
    cost: float
    observation_likelihood: dict[str, dict[str, float]]


DYNAMIC_PRIOR = {"correct": 0.70, "interface": 0.15, "boundary": 0.15}
DYNAMIC_QUERIES = {
    "triage_probe": Query(
        "triage_probe",
        0.02,
        {
            "interface_signal": {"correct": 0.50, "interface": 0.85, "boundary": 0.15},
            "boundary_signal": {"correct": 0.50, "interface": 0.15, "boundary": 0.85},
        },
    ),
    "interface_check": Query(
        "interface_check",
        0.04,
        {
            "pass": {"correct": 0.98, "interface": 0.01, "boundary": 0.60},
            "fail": {"correct": 0.02, "interface": 0.99, "boundary": 0.40},
        },
    ),
    "boundary_check": Query(
        "boundary_check",
        0.04,
        {
            "pass": {"correct": 0.98, "interface": 0.60, "boundary": 0.01},
            "fail": {"correct": 0.02, "interface": 0.40, "boundary": 0.99},
        },
    ),
    "broad_check": Query(
        "broad_check",
        0.06,
        {
            "pass": {"correct": 0.94, "interface": 0.10, "boundary": 0.10},
            "fail": {"correct": 0.06, "interface": 0.90, "boundary": 0.90},
        },
    ),
    "repeat_review": Query(
        "repeat_review",
        0.01,
        {"pass": {"correct": 1.00, "interface": 1.00, "boundary": 1.00}},
    ),
}
DYNAMIC_BUDGET = 0.095
DYNAMIC_TASKS = 100_000

COUPLED_BUDGET = 0.110
COUPLED_TASKS_PER_REP = 1000
COUPLED_HISTORY_MASS = {
    "ambiguous": 0.40,
    "interface_hint": 0.25,
    "boundary_hint": 0.25,
    "low_risk": 0.10,
}
COUPLED_TRUE_BELIEFS = {
    "ambiguous": {"correct": 0.76, "interface": 0.12, "boundary": 0.12},
    "interface_hint": {"correct": 0.76, "interface": 0.23, "boundary": 0.01},
    "boundary_hint": {"correct": 0.76, "interface": 0.01, "boundary": 0.23},
    "low_risk": {"correct": 0.90, "interface": 0.05, "boundary": 0.05},
}
COUPLED_CREDAL_BELIEFS = {
    "ambiguous": (
        {"correct": 0.76, "interface": 0.22, "boundary": 0.02},
        {"correct": 0.76, "interface": 0.02, "boundary": 0.22},
    ),
    "interface_hint": (
        {"correct": 0.76, "interface": 0.23, "boundary": 0.01},
        {"correct": 0.76, "interface": 0.17, "boundary": 0.07},
    ),
    "boundary_hint": (
        {"correct": 0.76, "interface": 0.01, "boundary": 0.23},
        {"correct": 0.76, "interface": 0.07, "boundary": 0.17},
    ),
    "low_risk": (
        {"correct": 0.90, "interface": 0.07, "boundary": 0.03},
        {"correct": 0.90, "interface": 0.03, "boundary": 0.07},
    ),
}
COUPLED_QUERIES = {
    "triage_probe": Query(
        "triage_probe",
        0.02,
        {
            "interface_signal": {"correct": 0.50, "interface": 0.95, "boundary": 0.05},
            "boundary_signal": {"correct": 0.50, "interface": 0.05, "boundary": 0.95},
        },
    ),
    "interface_check": Query(
        "interface_check",
        0.04,
        {
            "pass": {"correct": 0.98, "interface": 0.01, "boundary": 0.60},
            "fail": {"correct": 0.02, "interface": 0.99, "boundary": 0.40},
        },
    ),
    "boundary_check": Query(
        "boundary_check",
        0.04,
        {
            "pass": {"correct": 0.98, "interface": 0.60, "boundary": 0.01},
            "fail": {"correct": 0.02, "interface": 0.40, "boundary": 0.99},
        },
    ),
    "broad_check": Query(
        "broad_check",
        0.06,
        {
            "pass": {"correct": 0.94, "interface": 0.12, "boundary": 0.12},
            "fail": {"correct": 0.06, "interface": 0.88, "boundary": 0.88},
        },
    ),
    "full_check": Query(
        "full_check",
        0.08,
        {
            "pass": {"correct": 0.99, "interface": 0.005, "boundary": 0.005},
            "fail": {"correct": 0.01, "interface": 0.995, "boundary": 0.995},
        },
    ),
}

NECESSITY_HISTORY_MASS = {
    "interface_context": 0.40,
    "boundary_context": 0.40,
    "ambiguous_context": 0.20,
}
NECESSITY_TRUE_BELIEFS = {
    "interface_context": {"correct": 0.84, "interface": 0.14, "boundary": 0.02},
    "boundary_context": {"correct": 0.84, "interface": 0.02, "boundary": 0.14},
    "ambiguous_context": {"correct": 0.84, "interface": 0.08, "boundary": 0.08},
}
NECESSITY_CREDAL_BELIEFS = {
    "interface_context": (
        {"correct": 0.84, "interface": 0.15, "boundary": 0.01},
        {"correct": 0.84, "interface": 0.11, "boundary": 0.05},
    ),
    "boundary_context": (
        {"correct": 0.84, "interface": 0.01, "boundary": 0.15},
        {"correct": 0.84, "interface": 0.05, "boundary": 0.11},
    ),
    "ambiguous_context": (
        {"correct": 0.84, "interface": 0.14, "boundary": 0.02},
        {"correct": 0.84, "interface": 0.02, "boundary": 0.14},
    ),
}
NECESSITY_QUERIES = {
    "interface_check": Query(
        "interface_check",
        0.025,
        {
            "pass": {"correct": 0.995, "interface": 0.002, "boundary": 0.40},
            "fail": {"correct": 0.005, "interface": 0.998, "boundary": 0.60},
        },
    ),
    "boundary_check": Query(
        "boundary_check",
        0.025,
        {
            "pass": {"correct": 0.995, "interface": 0.40, "boundary": 0.002},
            "fail": {"correct": 0.005, "interface": 0.60, "boundary": 0.998},
        },
    ),
    "broad_check": Query(
        "broad_check",
        0.045,
        {
            "pass": {"correct": 0.995, "interface": 0.18, "boundary": 0.18},
            "fail": {"correct": 0.005, "interface": 0.82, "boundary": 0.82},
        },
    ),
    "full_check": Query(
        "full_check",
        0.080,
        {
            "pass": {"correct": 0.999, "interface": 0.003, "boundary": 0.003},
            "fail": {"correct": 0.001, "interface": 0.997, "boundary": 0.997},
        },
    ),
}
NECESSITY_BUDGET = 0.075
NECESSITY_DECLINE_LOSS = 0.25
NECESSITY_TASKS = 200_000


def binomial(rng: random.Random, n: int, p: float) -> int:
    return sum(rng.random() < p for _ in range(n))


def wilson_upper(successes: int, n: int, z: float = 1.6448536269514722) -> float:
    """One-sided Wilson upper bound with z=1.64485 for 95% one-sided."""
    if n <= 0:
        return 1.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return min(1.0, (centre + radius) / denom)


def wilson_interval(successes: int, n: int, z: float = 1.6448536269514722) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - radius) / denom), min(1.0, (centre + radius) / denom)


def normal_beta_upper(successes: int, n: int, z: float = 1.6448536269514722) -> float:
    """Approximate one-sided Beta(1,1) posterior upper quantile."""
    a = successes + 1.0
    b = n - successes + 1.0
    mean_value = a / (a + b)
    variance = a * b / ((a + b) ** 2 * (a + b + 1.0))
    return min(1.0, mean_value + z * math.sqrt(variance))


def false_independent_rate(common_miss: float) -> float:
    if common_miss >= FALSE_PASS_MARGINAL:
        return 0.0
    return (FALSE_PASS_MARGINAL - common_miss) / (1.0 - common_miss)


def pass3_given_error(common_miss: float) -> float:
    common_miss = min(max(common_miss, 0.0), FALSE_PASS_MARGINAL)
    fp = false_independent_rate(common_miss)
    return common_miss + (1.0 - common_miss) * fp**3


def risk_after_passes(common_miss: float, passes: int = 3) -> float:
    if passes == 1:
        p_pass_error = FALSE_PASS_MARGINAL
        p_pass_correct = TRUE_PASS
    elif passes == 3:
        p_pass_error = pass3_given_error(common_miss)
        p_pass_correct = TRUE_PASS**3
    else:
        raise ValueError("only one-pass and three-pass summaries are used")
    bad = (1.0 - 0.50) * p_pass_error
    good = 0.50 * p_pass_correct
    return bad / (bad + good)


def risk_after_orthogonal(common_miss: float, passes: int = 3) -> float:
    if passes == 1:
        p_pass_error = FALSE_PASS_MARGINAL
        p_pass_correct = TRUE_PASS
    elif passes == 3:
        p_pass_error = pass3_given_error(common_miss)
        p_pass_correct = TRUE_PASS**3
    else:
        raise ValueError("only one-pass and three-pass summaries are used")
    bad = (1.0 - 0.50) * p_pass_error * ORTHOGONAL_FALSE_PASS
    good = 0.50 * p_pass_correct * ORTHOGONAL_TRUE_PASS
    return bad / (bad + good)


def draw_group(rng: random.Random, groups: tuple[DependenceGroup, ...]) -> DependenceGroup:
    marker = rng.random()
    cumulative = 0.0
    for group in groups:
        cumulative += group.mass
        if marker <= cumulative:
            return group
    return groups[-1]


def cheap_validators_pass(rng: random.Random, *, correct: bool, common_miss: float) -> bool:
    if correct:
        return all(rng.random() < TRUE_PASS for _ in range(3))
    if rng.random() < common_miss:
        return True
    fp = false_independent_rate(common_miss)
    return all(rng.random() < fp for _ in range(3))


def orthogonal_passes(rng: random.Random, *, correct: bool) -> bool:
    p = ORTHOGONAL_TRUE_PASS if correct else ORTHOGONAL_FALSE_PASS
    return rng.random() < p


def dependence_decide(
    *,
    policy: str,
    common_miss_estimate: float,
    common_miss_upper: float,
    bayes_upper: float,
    correct: bool,
    cheap_pass: bool,
    orthogonal_pass: bool,
) -> tuple[str, float]:
    cost = 3.0 * CHEAP_CHECK_COST
    if policy == "abstain_all":
        return "abstain", cost
    if policy == "always_orthogonal":
        gamma = common_miss_upper
        cost += ORTHOGONAL_CHECK_COST
        if not cheap_pass or not orthogonal_pass:
            return "abstain", cost
        return ("accept" if risk_after_orthogonal(gamma) <= DELTA else "abstain"), cost
    if not cheap_pass:
        return "abstain", cost
    if policy == "independent_bayes":
        gamma = 0.0
    elif policy == "correlated_plugin":
        gamma = common_miss_estimate
    elif policy == "bayesian_upper":
        gamma = bayes_upper
    elif policy == "credal_dependence":
        gamma = common_miss_upper
    elif policy == "dedup_then_orthogonal":
        gamma = 0.0
        if risk_after_passes(gamma, passes=1) <= DELTA:
            return "accept", cost
        cost += ORTHOGONAL_CHECK_COST
        if not orthogonal_pass:
            return "abstain", cost
        return "accept", cost
    else:
        raise ValueError(policy)
    gamma = min(max(gamma, 0.0), FALSE_PASS_MARGINAL)
    if risk_after_passes(gamma) <= DELTA:
        return "accept", cost
    cost += ORTHOGONAL_CHECK_COST
    if not orthogonal_pass:
        return "abstain", cost
    return ("accept" if risk_after_orthogonal(gamma) <= DELTA else "abstain"), cost


def summarize_counts(rows: list[dict[str, float | bool | str]]) -> dict[str, float | int | None]:
    accepted = [row for row in rows if row["accepted"]]
    return {
        "tasks": len(rows),
        "accept_rate": mean(bool(row["accepted"]) for row in rows),
        "wrong_accept_per_task": mean(bool(row["false_accept"]) for row in rows),
        "error_among_accepted": (
            mean(bool(row["false_accept"]) for row in accepted) if accepted else None
        ),
        "correct_completion_rate": mean(bool(row["correct_completion"]) for row in rows),
        "abstain_rate": mean(str(row["action"]) == "abstain" for row in rows),
        "mean_cost": mean(float(row["cost"]) for row in rows),
    }


def run_dependence_same_distribution(
    rng: random.Random,
    *,
    calibration_reps: int,
    test_tasks_per_rep: int,
) -> dict[str, object]:
    per_rep: list[dict[str, object]] = []
    coverage_events = []
    for rep in range(calibration_reps):
        calibration = {
            group.name: binomial(rng, group.calibration_n, group.true_common_miss)
            for group in DEPENDENCE_GROUPS
        }
        group_bounds = {
            group.name: {
                "plugin": calibration[group.name] / group.calibration_n,
                "wilson": wilson_upper(calibration[group.name], group.calibration_n),
                "bayes": normal_beta_upper(calibration[group.name], group.calibration_n),
            }
            for group in DEPENDENCE_GROUPS
        }
        coverage_events.extend(
            group.true_common_miss <= group_bounds[group.name]["wilson"]
            for group in DEPENDENCE_GROUPS
        )
        policy_rows = {policy: [] for policy in DEPENDENCE_POLICIES}
        for _ in range(test_tasks_per_rep):
            group = draw_group(rng, DEPENDENCE_GROUPS)
            correct = rng.random() < 0.50
            cheap_pass = cheap_validators_pass(
                rng, correct=correct, common_miss=group.true_common_miss
            )
            orthogonal_pass = orthogonal_passes(rng, correct=correct)
            bounds = group_bounds[group.name]
            for policy in DEPENDENCE_POLICIES:
                action, cost = dependence_decide(
                    policy=policy,
                    common_miss_estimate=bounds["plugin"],
                    common_miss_upper=bounds["wilson"],
                    bayes_upper=bounds["bayes"],
                    correct=correct,
                    cheap_pass=cheap_pass,
                    orthogonal_pass=orthogonal_pass,
                )
                policy_rows[policy].append({
                    "action": action,
                    "cost": cost,
                    "accepted": action == "accept",
                    "false_accept": action == "accept" and not correct,
                    "correct_completion": action == "accept" and correct,
                })
        per_rep.append({
            "rep": rep,
            "summary": {
                policy: summarize_counts(rows)
                for policy, rows in policy_rows.items()
            },
        })
    aggregate = {}
    for policy in DEPENDENCE_POLICIES:
        aggregate[policy] = {}
        for metric in (
            "accept_rate",
            "wrong_accept_per_task",
            "correct_completion_rate",
            "abstain_rate",
            "mean_cost",
        ):
            aggregate[policy][metric] = mean(
                float(row["summary"][policy][metric]) for row in per_rep
            )
        error_values = [
            row["summary"][policy]["error_among_accepted"]
            for row in per_rep
            if row["summary"][policy]["error_among_accepted"] is not None
        ]
        aggregate[policy]["error_among_accepted"] = (
            mean(float(value) for value in error_values) if error_values else None
        )
    return {
        "calibration_reps": calibration_reps,
        "test_tasks_per_rep": test_tasks_per_rep,
        "wilson_group_coverage": mean(coverage_events),
        "summary": aggregate,
    }


def normalize(belief: dict[str, float]) -> dict[str, float]:
    total = sum(belief.values())
    return {state: value / total for state, value in belief.items()}


def draw_state(rng: random.Random) -> str:
    marker = rng.random()
    cumulative = 0.0
    for state, probability in DYNAMIC_PRIOR.items():
        cumulative += probability
        if marker <= cumulative:
            return state
    return "boundary"


def draw_from_distribution(rng: random.Random, distribution: dict[str, float]) -> str:
    marker = rng.random()
    cumulative = 0.0
    last = next(iter(distribution))
    for state, probability in distribution.items():
        cumulative += probability
        last = state
        if marker <= cumulative:
            return state
    return last


def error_risk(belief: dict[str, float]) -> float:
    return belief["interface"] + belief["boundary"]


def terminal_loss(belief: dict[str, float]) -> float:
    accept_loss = error_risk(belief) if error_risk(belief) <= DELTA else math.inf
    return min(accept_loss, DECLINE_LOSS)


def update_belief(
    belief: dict[str, float],
    query_name: str,
    observation: str,
    queries: dict[str, Query] = DYNAMIC_QUERIES,
) -> dict[str, float]:
    query = queries[query_name]
    weighted = {}
    for state, probability in belief.items():
        weighted[state] = probability * query.observation_likelihood[observation][state]
    return normalize(weighted)


def observation_probabilities(
    belief: dict[str, float],
    query_name: str,
    queries: dict[str, Query] = DYNAMIC_QUERIES,
) -> dict[str, float]:
    query = queries[query_name]
    return {
        observation: sum(
            probability * likelihood[state]
            for state, probability in belief.items()
        )
        for observation, likelihood in query.observation_likelihood.items()
    }


def draw_query_observation(
    rng: random.Random,
    state: str,
    query_name: str,
    queries: dict[str, Query] = DYNAMIC_QUERIES,
) -> str:
    query = queries[query_name]
    marker = rng.random()
    cumulative = 0.0
    observation = next(iter(query.observation_likelihood))
    for name, likelihood in query.observation_likelihood.items():
        cumulative += likelihood[state]
        if marker <= cumulative:
            observation = name
            break
    return observation


def query_expected_loss(belief: dict[str, float], query_name: str) -> float:
    query = DYNAMIC_QUERIES[query_name]
    total = query.cost
    for observation, observation_probability in observation_probabilities(belief, query_name).items():
        if observation_probability > 0:
            total += observation_probability * terminal_loss(
                update_belief(belief, query_name, observation)
            )
    return total


def best_future_loss(
    belief: dict[str, float],
    used: set[str],
    remaining: float,
    depth: int,
    queries: dict[str, Query] = DYNAMIC_QUERIES,
) -> float:
    best = terminal_loss(belief)
    if depth <= 0:
        return best
    for query_name, query in queries.items():
        if query_name in used or query.cost > remaining:
            continue
        expected = query.cost
        for observation, probability in observation_probabilities(belief, query_name, queries).items():
            expected += probability * best_future_loss(
                update_belief(belief, query_name, observation, queries),
                used | {query_name},
                remaining - query.cost,
                depth - 1,
                queries,
            )
        best = min(best, expected)
    return best


def best_value_query(
    belief: dict[str, float],
    used: set[str],
    remaining: float,
    depth: int,
    queries: dict[str, Query] = DYNAMIC_QUERIES,
) -> tuple[str | None, float]:
    base = terminal_loss(belief)
    best_query = None
    best_value = 0.0
    for query_name, query in queries.items():
        if query_name in used or query.cost > remaining:
            continue
        expected = query.cost
        for observation, probability in observation_probabilities(belief, query_name, queries).items():
            expected += probability * best_future_loss(
                update_belief(belief, query_name, observation, queries),
                used | {query_name},
                remaining - query.cost,
                depth - 1,
                queries,
            )
        value = base - expected
        if value > best_value:
            best_query = query_name
            best_value = value
    return best_query, best_value


def run_dynamic_policy(
    *,
    policy: str,
    state: str,
    observations: dict[str, str],
) -> dict[str, object]:
    belief = dict(DYNAMIC_PRIOR)
    cost = 3.0 * CHEAP_CHECK_COST
    used: list[str] = []
    for step in range(2):
        if error_risk(belief) <= DELTA:
            break
        remaining = DYNAMIC_BUDGET - cost
        if remaining <= 0:
            break
        if policy == "dynamic_credal":
            query_name, value = best_value_query(belief, set(used), remaining, depth=2)
            if query_name is None or value <= 0.0:
                break
        elif policy == "bayesian_depth2":
            query_name, value = best_value_query(belief, set(used), remaining, depth=2)
            if query_name is None or value <= 0.0:
                break
        elif policy == "one_step_credal":
            query_name, value = best_value_query(belief, set(used), remaining, depth=1)
            if query_name is None or value <= 0.0:
                break
        elif policy == "dynamic_no_update":
            query_name, value = best_value_query(dict(DYNAMIC_PRIOR), set(used), remaining, depth=2)
            if query_name is None or value <= 0.0:
                break
        elif policy == "fixed_triage_targeted":
            if step == 0:
                query_name = "triage_probe"
            elif observations["triage_probe"] == "interface_signal":
                query_name = "interface_check"
            else:
                query_name = "boundary_check"
        elif policy == "fixed_interface_first":
            order = ("interface_check", "boundary_check")
            query_name = order[step]
        elif policy == "fixed_boundary_first":
            order = ("boundary_check", "interface_check")
            query_name = order[step]
        elif policy == "fixed_broad":
            query_name = "broad_check" if step == 0 else "repeat_review"
        elif policy == "repeat_review":
            query_name = "repeat_review"
        else:
            raise ValueError(policy)
        query = DYNAMIC_QUERIES[query_name]
        if query.cost > remaining:
            break
        cost += query.cost
        used.append(query_name)
        observation = observations[query_name]
        belief = update_belief(belief, query_name, observation)
    accepted = error_risk(belief) <= DELTA
    return {
        "accepted": accepted,
        "false_accept": accepted and state != "correct",
        "correct_completion": accepted and state == "correct",
        "cost": cost,
        "first_triage": bool(used) and used[0] == "triage_probe",
        "first_interface": bool(used) and used[0] == "interface_check",
        "first_boundary": bool(used) and used[0] == "boundary_check",
        "repeat_bought": "repeat_review" in used,
        "checks": len(used),
    }


def run_dynamic_checks(rng: random.Random, *, tasks: int) -> dict[str, object]:
    policies = (
        "dynamic_credal",
        "bayesian_depth2",
        "fixed_triage_targeted",
        "one_step_credal",
        "dynamic_no_update",
        "fixed_interface_first",
        "fixed_boundary_first",
        "fixed_broad",
        "repeat_review",
    )
    rows = {policy: [] for policy in policies}
    for _ in range(tasks):
        state = draw_state(rng)
        observations = {
            query_name: draw_query_observation(rng, state, query_name)
            for query_name in DYNAMIC_QUERIES
        }
        for policy in policies:
            rows[policy].append(
                run_dynamic_policy(policy=policy, state=state, observations=observations)
            )
    summary = {}
    for policy, policy_rows in rows.items():
        base = summarize_counts([
            {
                "action": "accept" if row["accepted"] else "abstain",
                "cost": row["cost"],
                "accepted": row["accepted"],
                "false_accept": row["false_accept"],
                "correct_completion": row["correct_completion"],
            }
            for row in policy_rows
        ])
        base["first_triage_rate"] = mean(bool(row["first_triage"]) for row in policy_rows)
        base["first_interface_rate"] = mean(bool(row["first_interface"]) for row in policy_rows)
        base["first_boundary_rate"] = mean(bool(row["first_boundary"]) for row in policy_rows)
        base["repeat_check_rate"] = mean(bool(row["repeat_bought"]) for row in policy_rows)
        base["mean_checks"] = mean(int(row["checks"]) for row in policy_rows)
        summary[policy] = base
    return {
        "tasks": tasks,
        "prior": dict(DYNAMIC_PRIOR),
        "budget": DYNAMIC_BUDGET,
        "decline_loss": DECLINE_LOSS,
        "queries": {
            name: {
                "cost": query.cost,
                "observation_likelihood": {
                    observation: dict(likelihood)
                    for observation, likelihood in query.observation_likelihood.items()
                },
            }
            for name, query in DYNAMIC_QUERIES.items()
        },
        "summary": summary,
    }


def terminal_loss_set(beliefs: tuple[dict[str, float], ...]) -> float:
    upper_risk = max(error_risk(belief) for belief in beliefs)
    accept_loss = upper_risk if upper_risk <= DELTA else math.inf
    return min(accept_loss, DECLINE_LOSS)


def update_belief_set(
    beliefs: tuple[dict[str, float], ...],
    query_name: str,
    observation: str,
    queries: dict[str, Query],
) -> tuple[dict[str, float], ...]:
    return tuple(update_belief(belief, query_name, observation, queries) for belief in beliefs)


def robust_future_loss(
    beliefs: tuple[dict[str, float], ...],
    used: set[str],
    remaining: float,
    depth: int,
    queries: dict[str, Query],
) -> float:
    best = terminal_loss_set(beliefs)
    if depth <= 0:
        return best
    for query_name, query in queries.items():
        if query_name in used or query.cost > remaining:
            continue
        future_values = {
            observation: robust_future_loss(
                update_belief_set(beliefs, query_name, observation, queries),
                used | {query_name},
                remaining - query.cost,
                depth - 1,
                queries,
            )
            for observation in query.observation_likelihood
        }
        worst_expectation = max(
            sum(
                observation_probabilities(belief, query_name, queries)[observation]
                * future_values[observation]
                for observation in future_values
            )
            for belief in beliefs
        )
        best = min(best, query.cost + worst_expectation)
    return best


def robust_value_query(
    beliefs: tuple[dict[str, float], ...],
    used: set[str],
    remaining: float,
    depth: int,
    queries: dict[str, Query],
) -> tuple[str | None, float]:
    base = terminal_loss_set(beliefs)
    best_query = None
    best_value = 0.0
    for query_name, query in queries.items():
        if query_name in used or query.cost > remaining:
            continue
        future_values = {
            observation: robust_future_loss(
                update_belief_set(beliefs, query_name, observation, queries),
                used | {query_name},
                remaining - query.cost,
                depth - 1,
                queries,
            )
            for observation in query.observation_likelihood
        }
        worst_expectation = max(
            sum(
                observation_probabilities(belief, query_name, queries)[observation]
                * future_values[observation]
                for observation in future_values
            )
            for belief in beliefs
        )
        value = base - (query.cost + worst_expectation)
        if value > best_value:
            best_query = query_name
            best_value = value
    return best_query, best_value


def coupled_belief_set(
    history_name: str,
    calibration_counts: dict[str, tuple[int, int]],
) -> tuple[dict[str, float], ...]:
    interface_count, n = calibration_counts[history_name]
    lower, upper = wilson_interval(interface_count, n)
    total_error = 1.0 - COUPLED_TRUE_BELIEFS[history_name]["correct"]
    correct = 1.0 - total_error
    return (
        {"correct": correct, "interface": total_error * lower, "boundary": total_error * (1.0 - lower)},
        {"correct": correct, "interface": total_error * upper, "boundary": total_error * (1.0 - upper)},
    )


def coupled_plugin_belief(
    history_name: str,
    calibration_counts: dict[str, tuple[int, int]],
) -> dict[str, float]:
    interface_count, n = calibration_counts[history_name]
    p_interface = interface_count / n if n else 0.5
    total_error = 1.0 - COUPLED_TRUE_BELIEFS[history_name]["correct"]
    correct = 1.0 - total_error
    return {
        "correct": correct,
        "interface": total_error * p_interface,
        "boundary": total_error * (1.0 - p_interface),
    }


def coupled_static_worst_belief(
    beliefs: tuple[dict[str, float], ...],
) -> dict[str, float]:
    return max(beliefs, key=lambda belief: (belief["interface"], error_risk(belief)))


def run_coupled_policy(
    *,
    policy: str,
    history_name: str,
    state: str,
    observations: dict[str, str],
    calibration_counts: dict[str, tuple[int, int]],
) -> dict[str, object]:
    if policy == "bayesian_depth2":
        beliefs = (coupled_plugin_belief(history_name, calibration_counts),)
        value_rule = "point"
    elif policy == "static_worst_depth2":
        beliefs = (coupled_static_worst_belief(coupled_belief_set(history_name, calibration_counts)),)
        value_rule = "point"
    else:
        beliefs = coupled_belief_set(history_name, calibration_counts)
        value_rule = "credal"
    cost = 3.0 * CHEAP_CHECK_COST
    used: list[str] = []
    for step in range(2):
        if terminal_loss_set(beliefs) < DECLINE_LOSS:
            break
        remaining = COUPLED_BUDGET - cost
        if remaining <= 0:
            break
        if policy == "dynamic_credal_depth2":
            query_name, value = robust_value_query(
                beliefs, set(used), remaining, depth=2, queries=COUPLED_QUERIES
            )
            if query_name is None or value <= 0.0:
                break
        elif policy in {"bayesian_depth2", "static_worst_depth2"}:
            query_name, value = best_value_query(
                beliefs[0], set(used), remaining, depth=2, queries=COUPLED_QUERIES
            )
            if query_name is None or value <= 0.0:
                break
        elif policy == "fixed_triage_targeted":
            if step == 0:
                query_name = "triage_probe"
            elif observations["triage_probe"] == "interface_signal":
                query_name = "interface_check"
            else:
                query_name = "boundary_check"
        elif policy == "fixed_broad":
            query_name = "broad_check" if step == 0 else "triage_probe"
        elif policy == "full_check":
            query_name = "full_check" if step == 0 else "triage_probe"
        else:
            raise ValueError(policy)
        query = COUPLED_QUERIES[query_name]
        if query.cost > remaining:
            break
        cost += query.cost
        used.append(query_name)
        beliefs = update_belief_set(beliefs, query_name, observations[query_name], COUPLED_QUERIES)
    accepted = terminal_loss_set(beliefs) < DECLINE_LOSS
    return {
        "accepted": accepted,
        "false_accept": accepted and state != "correct",
        "correct_completion": accepted and state == "correct",
        "cost": cost,
        "first_query": used[0] if used else "none",
        "route": "->".join(used) if used else "none",
        "first_triage": bool(used) and used[0] == "triage_probe",
        "first_specific": bool(used) and used[0] in {"interface_check", "boundary_check"},
        "first_full": bool(used) and used[0] == "full_check",
        "checks": len(used),
        "value_rule": value_rule,
    }


def sample_coupled_calibration(rng: random.Random) -> dict[str, tuple[int, int]]:
    sample_sizes = {
        "ambiguous": 80,
        "interface_hint": 80,
        "boundary_hint": 80,
        "low_risk": 120,
    }
    counts = {}
    for history_name, n in sample_sizes.items():
        truth = COUPLED_TRUE_BELIEFS[history_name]
        error_mass = truth["interface"] + truth["boundary"]
        p_interface = truth["interface"] / error_mass if error_mass else 0.5
        counts[history_name] = (binomial(rng, n, p_interface), n)
    return counts


def run_coupled_active_experiment(
    rng: random.Random,
    *,
    calibration_reps: int,
    test_tasks_per_rep: int,
) -> dict[str, object]:
    policies = (
        "dynamic_credal_depth2",
        "bayesian_depth2",
        "fixed_triage_targeted",
        "static_worst_depth2",
        "fixed_broad",
        "full_check",
    )
    per_rep = []
    disagreement = _new_coupled_disagreement_accumulator()
    first_query_by_history = {policy: {name: [] for name in COUPLED_HISTORY_MASS} for policy in policies}
    for rep in range(calibration_reps):
        calibration_counts = sample_coupled_calibration(rng)
        rows = {policy: [] for policy in policies}
        decision_cache: dict[tuple[str, str, tuple[tuple[str, str], ...]], dict[str, object]] = {}
        for _ in range(test_tasks_per_rep):
            history_name = draw_from_distribution(rng, COUPLED_HISTORY_MASS)
            state = draw_from_distribution(rng, COUPLED_TRUE_BELIEFS[history_name])
            observations = {
                query_name: draw_query_observation(rng, state, query_name, COUPLED_QUERIES)
                for query_name in COUPLED_QUERIES
            }
            observation_key = tuple(sorted(observations.items()))
            task_results: dict[str, dict[str, object]] = {}
            for policy in policies:
                cache_key = (policy, history_name, observation_key)
                if cache_key not in decision_cache:
                    decision_cache[cache_key] = run_coupled_policy(
                        policy=policy,
                        history_name=history_name,
                        state="correct",
                        observations=observations,
                        calibration_counts=calibration_counts,
                    )
                cached = decision_cache[cache_key]
                result = {
                    **cached,
                    "false_accept": bool(cached["accepted"]) and state != "correct",
                    "correct_completion": bool(cached["accepted"]) and state == "correct",
                }
                task_results[policy] = result
                rows[policy].append({
                    "action": "accept" if result["accepted"] else "abstain",
                    **result,
                })
                if result["checks"]:
                    if result["first_triage"]:
                        first = "triage"
                    elif result["first_specific"]:
                        first = "specific"
                    elif result["first_full"]:
                        first = "full"
                    else:
                        first = "other"
                else:
                    first = "none"
                first_query_by_history[policy][history_name].append(first)
            _record_coupled_disagreement(
                disagreement,
                history_name=history_name,
                state=state,
                calibration_counts=calibration_counts,
                task_results=task_results,
            )
        per_rep.append({
            "rep": rep,
            "summary": {policy: summarize_counts(rows[policy]) for policy in policies},
            "first_query": {
                policy: {
                    history_name: {
                        label: first_query_by_history[policy][history_name].count(label)
                        / len(first_query_by_history[policy][history_name])
                        for label in ("triage", "specific", "full", "none", "other")
                    }
                    for history_name in COUPLED_HISTORY_MASS
                    if first_query_by_history[policy][history_name]
                }
                for policy in policies
            },
        })
    aggregate = {}
    for policy in policies:
        aggregate[policy] = {}
        for metric in (
            "accept_rate",
            "wrong_accept_per_task",
            "correct_completion_rate",
            "abstain_rate",
            "mean_cost",
        ):
            values = [float(row["summary"][policy][metric]) for row in per_rep]
            aggregate[policy][metric] = mean(values)
            aggregate[policy][metric + "_min"] = min(values)
            aggregate[policy][metric + "_max"] = max(values)
        error_values = [
            row["summary"][policy]["error_among_accepted"]
            for row in per_rep
            if row["summary"][policy]["error_among_accepted"] is not None
        ]
        aggregate[policy]["error_among_accepted"] = (
            mean(float(value) for value in error_values) if error_values else None
        )
    paired_differences = coupled_paired_differences(per_rep)
    return {
        "calibration_reps": calibration_reps,
        "test_tasks_per_rep": test_tasks_per_rep,
        "budget": COUPLED_BUDGET,
        "history_mass": dict(COUPLED_HISTORY_MASS),
        "true_beliefs": {key: dict(value) for key, value in COUPLED_TRUE_BELIEFS.items()},
        "queries": {
            name: {
                "cost": query.cost,
                "observation_likelihood": {
                    observation: dict(likelihood)
                    for observation, likelihood in query.observation_likelihood.items()
                },
            }
            for name, query in COUPLED_QUERIES.items()
        },
        "summary": aggregate,
        "paired_differences": paired_differences,
        "paired_difference_stability": coupled_paired_difference_stability(per_rep),
        "disagreement_attribution": _summarize_coupled_disagreement(disagreement),
        "first_query_by_history": per_rep[-1]["first_query"],
        "per_rep_summary": per_rep,
    }


def percentile_interval(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    ordered = sorted(values)
    lo = ordered[int(0.025 * (len(ordered) - 1))]
    hi = ordered[int(0.975 * (len(ordered) - 1))]
    return (mean(values), lo, hi)


def coupled_paired_differences(per_rep: list[dict[str, object]]) -> dict[str, dict[str, tuple[float, float, float]]]:
    baselines = {
        "bayesian_depth2": "Bayesian plug-in depth 2",
        "static_worst_depth2": "Static worst-model depth 2",
        "fixed_broad": "Fixed broad check",
    }
    metrics = (
        "error_among_accepted",
        "wrong_accept_per_task",
        "correct_completion_rate",
        "mean_cost",
    )
    result: dict[str, dict[str, tuple[float, float, float]]] = {}
    for baseline in baselines:
        result[baseline] = {}
        for metric in metrics:
            values: list[float] = []
            for row in per_rep:
                summary = row["summary"]  # type: ignore[index]
                left = summary["dynamic_credal_depth2"][metric]  # type: ignore[index]
                right = summary[baseline][metric]  # type: ignore[index]
                if left is None or right is None:
                    continue
                values.append(float(left) - float(right))
            result[baseline][metric] = percentile_interval(values)
    return result


def _bootstrap_mean_interval(values: list[float], *, seed: int = 20261003, draws: int = 2000) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(draws):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return mean(values), means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))]


def coupled_paired_difference_stability(per_rep: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    baselines = ("bayesian_depth2", "static_worst_depth2", "fixed_broad")
    metrics = (
        "error_among_accepted",
        "wrong_accept_per_task",
        "correct_completion_rate",
        "mean_cost",
    )
    higher_is_better = {"correct_completion_rate"}
    result: dict[str, dict[str, dict[str, float]]] = {}
    for baseline in baselines:
        result[baseline] = {}
        for metric in metrics:
            values: list[float] = []
            for row in per_rep:
                summary = row["summary"]  # type: ignore[index]
                left = summary["dynamic_credal_depth2"][metric]  # type: ignore[index]
                right = summary[baseline][metric]  # type: ignore[index]
                if left is None or right is None:
                    continue
                values.append(float(left) - float(right))
            avg, boot_lo, boot_hi = _bootstrap_mean_interval(values)
            _, central_lo, central_hi = percentile_interval(values)
            if metric in higher_is_better:
                improved = sum(value > 1e-12 for value in values)
                worsened = sum(value < -1e-12 for value in values)
            else:
                improved = sum(value < -1e-12 for value in values)
                worsened = sum(value > 1e-12 for value in values)
            tied = len(values) - improved - worsened
            result[baseline][metric] = {
                "mean": avg,
                "bootstrap_lo": boot_lo,
                "bootstrap_hi": boot_hi,
                "central_lo": central_lo,
                "central_hi": central_hi,
                "improved": improved / len(values) if values else 0.0,
                "tied": tied / len(values) if values else 0.0,
                "worsened": worsened / len(values) if values else 0.0,
            }
    return result


def _new_coupled_disagreement_accumulator() -> dict[str, dict[str, dict[str, list[float]]]]:
    baselines = ("bayesian_depth2", "static_worst_depth2")
    histories = tuple(COUPLED_HISTORY_MASS) + ("all",)
    return {
        baseline: {
            history: {
                "tasks": [],
                "different": [],
                "dynamic_extra_triage": [],
                "wrong_accept_delta": [],
                "correct_completion_delta": [],
                "cost_delta": [],
                "avoided_wrong_accept": [],
                "lost_correct_accept": [],
                "plugin_fraction": [],
                "wilson_width": [],
            }
            for history in histories
        }
        for baseline in baselines
    }


def _record_coupled_disagreement(
    disagreement: dict[str, dict[str, dict[str, list[float]]]],
    *,
    history_name: str,
    state: str,
    calibration_counts: dict[str, tuple[int, int]],
    task_results: dict[str, dict[str, object]],
) -> None:
    del state
    dynamic = task_results["dynamic_credal_depth2"]
    interface_count, n = calibration_counts[history_name]
    lower, upper = wilson_interval(interface_count, n)
    plugin_fraction = interface_count / n if n else 0.5
    for baseline in ("bayesian_depth2", "static_worst_depth2"):
        base = task_results[baseline]
        first_differs = str(dynamic["first_query"]) != str(base["first_query"])
        route_differs = first_differs or bool(dynamic["accepted"]) != bool(base["accepted"])
        for bucket in (history_name, "all"):
            cells = disagreement[baseline][bucket]
            cells["tasks"].append(1.0)
            cells["different"].append(float(route_differs))
            if not route_differs:
                continue
            cells["dynamic_extra_triage"].append(float(dynamic["first_query"] == "triage_probe" and base["first_query"] != "triage_probe"))
            cells["wrong_accept_delta"].append(float(dynamic["false_accept"]) - float(base["false_accept"]))
            cells["correct_completion_delta"].append(float(dynamic["correct_completion"]) - float(base["correct_completion"]))
            cells["cost_delta"].append(float(dynamic["cost"]) - float(base["cost"]))
            cells["avoided_wrong_accept"].append(float(bool(base["false_accept"]) and not bool(dynamic["false_accept"])))
            cells["lost_correct_accept"].append(float(bool(base["correct_completion"]) and not bool(dynamic["correct_completion"])))
            cells["plugin_fraction"].append(plugin_fraction)
            cells["wilson_width"].append(upper - lower)


def _summarize_coupled_disagreement(
    disagreement: dict[str, dict[str, dict[str, list[float]]]]
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for baseline, histories in disagreement.items():
        summary[baseline] = {}
        for history, cells in histories.items():
            tasks = len(cells["tasks"])
            different = int(sum(cells["different"]))
            row = {
                "tasks": float(tasks),
                "different": float(different),
                "different_rate": different / tasks if tasks else 0.0,
            }
            for key in (
                "dynamic_extra_triage",
                "wrong_accept_delta",
                "correct_completion_delta",
                "cost_delta",
                "avoided_wrong_accept",
                "lost_correct_accept",
                "plugin_fraction",
                "wilson_width",
            ):
                values = cells[key]
                row[key] = mean(values) if values else 0.0
            summary[baseline][history] = row
    return summary


def necessity_terminal_loss_point(belief: dict[str, float]) -> float:
    risk = error_risk(belief)
    accept_loss = risk if risk <= DELTA else math.inf
    return min(accept_loss, NECESSITY_DECLINE_LOSS)


def necessity_terminal_loss_set(beliefs: tuple[dict[str, float], ...]) -> float:
    upper_risk = max(error_risk(belief) for belief in beliefs)
    accept_loss = upper_risk if upper_risk <= DELTA else math.inf
    return min(accept_loss, NECESSITY_DECLINE_LOSS)


def necessity_best_value_query(
    belief: dict[str, float],
    remaining: float,
) -> tuple[str | None, float]:
    base = necessity_terminal_loss_point(belief)
    best_query = None
    best_value = 0.0
    for query_name, query in NECESSITY_QUERIES.items():
        if query.cost > remaining:
            continue
        expected = query.cost
        for observation, probability in observation_probabilities(
            belief, query_name, NECESSITY_QUERIES
        ).items():
            expected += probability * necessity_terminal_loss_point(
                update_belief(belief, query_name, observation, NECESSITY_QUERIES)
            )
        value = base - expected
        if value > best_value:
            best_query = query_name
            best_value = value
    return best_query, best_value


def necessity_robust_value_query(
    beliefs: tuple[dict[str, float], ...],
    remaining: float,
) -> tuple[str | None, float]:
    base = necessity_terminal_loss_set(beliefs)
    best_query = None
    best_value = 0.0
    for query_name, query in NECESSITY_QUERIES.items():
        if query.cost > remaining:
            continue
        future_values = {
            observation: necessity_terminal_loss_set(
                update_belief_set(beliefs, query_name, observation, NECESSITY_QUERIES)
            )
            for observation in query.observation_likelihood
        }
        worst_expectation = max(
            sum(
                observation_probabilities(belief, query_name, NECESSITY_QUERIES)[observation]
                * future_values[observation]
                for observation in future_values
            )
            for belief in beliefs
        )
        value = base - (query.cost + worst_expectation)
        if value > best_value:
            best_query = query_name
            best_value = value
    return best_query, best_value


def necessity_accepts_point(
    belief: dict[str, float],
    query_name: str,
    observation: str,
) -> bool:
    updated = update_belief(belief, query_name, observation, NECESSITY_QUERIES)
    return error_risk(updated) <= DELTA


def necessity_accepts_credal(
    beliefs: tuple[dict[str, float], ...],
    query_name: str,
    observation: str,
) -> bool:
    updated = update_belief_set(beliefs, query_name, observation, NECESSITY_QUERIES)
    return necessity_terminal_loss_set(updated) < NECESSITY_DECLINE_LOSS


def run_necessity_policy(
    *,
    policy: str,
    history_name: str,
    state: str,
    observations: dict[str, str],
) -> dict[str, object]:
    remaining = NECESSITY_BUDGET - 3.0 * CHEAP_CHECK_COST
    cost = 3.0 * CHEAP_CHECK_COST
    if policy == "dynamic_credal":
        beliefs = NECESSITY_CREDAL_BELIEFS[history_name]
        query_name, value = necessity_robust_value_query(beliefs, remaining)
        if query_name is None or value <= 0:
            return {"accepted": False, "cost": cost, "first_query": "none"}
        cost += NECESSITY_QUERIES[query_name].cost
        accepted = necessity_accepts_credal(beliefs, query_name, observations[query_name])
    elif policy == "bayesian_history_plugin":
        belief = NECESSITY_TRUE_BELIEFS[history_name]
        query_name, value = necessity_best_value_query(belief, remaining)
        if query_name is None or value <= 0:
            return {"accepted": False, "cost": cost, "first_query": "none"}
        cost += NECESSITY_QUERIES[query_name].cost
        accepted = necessity_accepts_point(belief, query_name, observations[query_name])
    elif policy == "scalar_same_risk":
        belief = {"correct": 0.84, "interface": 0.08, "boundary": 0.08}
        query_name, value = necessity_best_value_query(belief, remaining)
        if query_name is None or value <= 0:
            return {"accepted": False, "cost": cost, "first_query": "none"}
        cost += NECESSITY_QUERIES[query_name].cost
        accepted = necessity_accepts_point(belief, query_name, observations[query_name])
    elif policy == "static_interface":
        belief = {"correct": 0.84, "interface": 0.15, "boundary": 0.01}
        query_name = "interface_check"
        cost += NECESSITY_QUERIES[query_name].cost
        accepted = necessity_accepts_point(belief, query_name, observations[query_name])
    elif policy == "static_boundary":
        belief = {"correct": 0.84, "interface": 0.01, "boundary": 0.15}
        query_name = "boundary_check"
        cost += NECESSITY_QUERIES[query_name].cost
        accepted = necessity_accepts_point(belief, query_name, observations[query_name])
    elif policy == "fixed_broad":
        belief = {"correct": 0.84, "interface": 0.08, "boundary": 0.08}
        query_name = "broad_check"
        cost += NECESSITY_QUERIES[query_name].cost
        accepted = necessity_accepts_point(belief, query_name, observations[query_name])
    elif policy == "full_check":
        belief = {"correct": 0.84, "interface": 0.08, "boundary": 0.08}
        query_name = "full_check"
        cost += NECESSITY_QUERIES[query_name].cost
        accepted = necessity_accepts_point(belief, query_name, observations[query_name])
    else:
        raise ValueError(policy)
    return {
        "accepted": accepted,
        "false_accept": accepted and state != "correct",
        "correct_completion": accepted and state == "correct",
        "cost": cost,
        "first_query": query_name,
    }


def run_credal_necessity_stress(rng: random.Random, *, tasks: int) -> dict[str, object]:
    policies = (
        "dynamic_credal",
        "bayesian_history_plugin",
        "scalar_same_risk",
        "static_interface",
        "static_boundary",
        "fixed_broad",
        "full_check",
    )
    rows = {policy: [] for policy in policies}
    first_query_by_history = {policy: {name: [] for name in NECESSITY_HISTORY_MASS} for policy in policies}
    for _ in range(tasks):
        history_name = draw_from_distribution(rng, NECESSITY_HISTORY_MASS)
        state = draw_from_distribution(rng, NECESSITY_TRUE_BELIEFS[history_name])
        observations = {
            query_name: draw_query_observation(rng, state, query_name, NECESSITY_QUERIES)
            for query_name in NECESSITY_QUERIES
        }
        for policy in policies:
            result = run_necessity_policy(
                policy=policy,
                history_name=history_name,
                state=state,
                observations=observations,
            )
            result.setdefault("false_accept", False)
            result.setdefault("correct_completion", False)
            rows[policy].append({
                "action": "accept" if result["accepted"] else "abstain",
                **result,
            })
            first_query_by_history[policy][history_name].append(str(result["first_query"]))
    summary = {}
    for policy, policy_rows in rows.items():
        base = summarize_counts(policy_rows)
        base["targeted_query_rate"] = mean(
            row["first_query"] in {"interface_check", "boundary_check"}
            for row in policy_rows
        )
        base["broad_query_rate"] = mean(row["first_query"] == "broad_check" for row in policy_rows)
        base["full_query_rate"] = mean(row["first_query"] == "full_check" for row in policy_rows)
        summary[policy] = base
    path_summary = {}
    for policy in policies:
        path_summary[policy] = {}
        for history_name in NECESSITY_HISTORY_MASS:
            choices = first_query_by_history[policy][history_name]
            path_summary[policy][history_name] = {
                choice: choices.count(choice) / len(choices)
                for choice in ("interface_check", "boundary_check", "broad_check", "full_check", "none")
            }
    return {
        "tasks": tasks,
        "budget": NECESSITY_BUDGET,
        "decline_loss": NECESSITY_DECLINE_LOSS,
        "history_mass": dict(NECESSITY_HISTORY_MASS),
        "true_beliefs": {key: dict(value) for key, value in NECESSITY_TRUE_BELIEFS.items()},
        "credal_beliefs": {
            key: [dict(row) for row in value]
            for key, value in NECESSITY_CREDAL_BELIEFS.items()
        },
        "queries": {
            name: {
                "cost": query.cost,
                "observation_likelihood": {
                    observation: dict(likelihood)
                    for observation, likelihood in query.observation_likelihood.items()
                },
            }
            for name, query in NECESSITY_QUERIES.items()
        },
        "summary": summary,
        "first_query_by_history": path_summary,
    }


def run_shift_stress(rng: random.Random, *, tasks: int) -> dict[str, object]:
    calibration_misses = 1
    calibration_n = 80
    true_common_miss = 0.18
    rows = {policy: [] for policy in DEPENDENCE_POLICIES}
    for _ in range(tasks):
        correct = rng.random() < 0.50
        cheap_pass = cheap_validators_pass(rng, correct=correct, common_miss=true_common_miss)
        orthogonal_pass = orthogonal_passes(rng, correct=correct)
        for policy in DEPENDENCE_POLICIES:
            action, cost = dependence_decide(
                policy=policy,
                common_miss_estimate=calibration_misses / calibration_n,
                common_miss_upper=wilson_upper(calibration_misses, calibration_n),
                bayes_upper=normal_beta_upper(calibration_misses, calibration_n),
                correct=correct,
                cheap_pass=cheap_pass,
                orthogonal_pass=orthogonal_pass,
            )
            rows[policy].append({
                "action": action,
                "cost": cost,
                "accepted": action == "accept",
                "false_accept": action == "accept" and not correct,
                "correct_completion": action == "accept" and correct,
            })
    return {
        "calibration_misses": calibration_misses,
        "calibration_n": calibration_n,
        "true_common_miss": true_common_miss,
        "wilson_upper": wilson_upper(calibration_misses, calibration_n),
        "bayesian_upper": normal_beta_upper(calibration_misses, calibration_n),
        "rare_under_iid_probability": sum(
            math.comb(calibration_n, k)
            * true_common_miss**k
            * (1.0 - true_common_miss) ** (calibration_n - k)
            for k in range(calibration_misses + 1)
        ),
        "summary": {policy: summarize_counts(policy_rows) for policy, policy_rows in rows.items()},
    }


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def write_dependence_latex(result: dict[str, object]) -> None:
    labels = {
        "independent_bayes": "Independent Bayes",
        "correlated_plugin": "Correlated plug-in",
        "bayesian_upper": "Bayesian upper",
        "credal_dependence": "Credal upper",
        "dedup_then_orthogonal": "Dedup + orthogonal",
        "always_orthogonal": "Always orthogonal",
        "abstain_all": "Abstain all",
    }
    summary = result["same_distribution"]["summary"]
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Policy & Accept & Error/accepted & Wrong accept & Correct completion & Cost\\",
        r"\midrule",
    ]
    for policy in DEPENDENCE_POLICIES:
        row = summary[policy]
        lines.append(
            "{} & {} & {} & {} & {} & {}\\\\".format(
                labels[policy],
                fmt(row["accept_rate"]),
                fmt(row["error_among_accepted"]),
                fmt(row["wrong_accept_per_task"]),
                fmt(row["correct_completion_rate"]),
                fmt(row["mean_cost"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    DEPENDENCE_TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_dynamic_latex(result: dict[str, object]) -> None:
    labels = {
        "dynamic_credal": "Depth-2 value planner",
        "bayesian_depth2": "Bayesian depth 2",
        "fixed_triage_targeted": "Fixed triage then target",
        "one_step_credal": "Depth-1 value planner",
        "dynamic_no_update": "No posterior update",
        "fixed_interface_first": "Single interface check",
        "fixed_boundary_first": "Single boundary check",
        "fixed_broad": "Fixed broad check",
        "repeat_review": "Repeat review",
    }
    summary = result["dynamic_checks"]["summary"]
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Policy & Accept & Error/accepted & Correct completion & Cost & First triage & Checks\\",
        r"\midrule",
    ]
    for policy in labels:
        row = summary[policy]
        lines.append(
            "{} & {} & {} & {} & {} & {} & {}\\\\".format(
                labels[policy],
                fmt(row["accept_rate"]),
                fmt(row["error_among_accepted"]),
                fmt(row["correct_completion_rate"]),
                fmt(row["mean_cost"]),
                fmt(row["first_triage_rate"]),
                fmt(row["mean_checks"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    DYNAMIC_TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_coupled_latex(result: dict[str, object]) -> None:
    labels = {
        "dynamic_credal_depth2": "Dynamic credal depth 2",
        "bayesian_depth2": "Bayesian depth 2",
        "fixed_triage_targeted": "Fixed triage rule",
        "static_worst_depth2": "Static worst-model depth 2",
        "fixed_broad": "Fixed broad check",
        "full_check": "Always full check",
    }
    summary = result["coupled_active"]["summary"]
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Policy & Accept & Error/accepted & Wrong accept & Correct completion & Cost & Cost range\\",
        r"\midrule",
    ]
    for policy in labels:
        row = summary[policy]
        cost_range = f"[{row['mean_cost_min']:.3f},{row['mean_cost_max']:.3f}]"
        lines.append(
            "{} & {} & {} & {} & {} & {} & {}\\\\".format(
                labels[policy],
                fmt(row["accept_rate"]),
                fmt(row["error_among_accepted"]),
                fmt(row["wrong_accept_per_task"]),
                fmt(row["correct_completion_rate"]),
                fmt(row["mean_cost"]),
                cost_range,
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    COUPLED_TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def _dominant_first_choice(first_query: dict[str, float]) -> str:
    label, value = max(first_query.items(), key=lambda item: item[1])
    names = {
        "triage": "triage",
        "specific": "specific",
        "full": "full",
        "none": "none",
        "other": "other",
    }
    return f"{names[label]} ({value:.2f})"


def write_coupled_path_latex(result: dict[str, object]) -> None:
    paths = result["coupled_active"]["first_query_by_history"]
    history_labels = {
        "ambiguous": "Ambiguous",
        "interface_hint": "Interface hint",
        "boundary_hint": "Boundary hint",
        "low_risk": "Lower risk",
    }
    lines = [
        r"\begin{tabular}{@{}llll@{}}",
        r"\toprule",
        r"History & Dynamic credal & Bayesian depth 2 & Static worst model\\",
        r"\midrule",
    ]
    for history_name, label in history_labels.items():
        lines.append(
            "{} & {} & {} & {}\\\\".format(
                label,
                _dominant_first_choice(paths["dynamic_credal_depth2"][history_name]),
                _dominant_first_choice(paths["bayesian_depth2"][history_name]),
                _dominant_first_choice(paths["static_worst_depth2"][history_name]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    COUPLED_PATH_TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_coupled_delta_latex(result: dict[str, object]) -> None:
    labels = {
        "bayesian_depth2": "Bayesian plug-in depth 2",
        "static_worst_depth2": "Static worst-model depth 2",
        "fixed_broad": "Fixed broad check",
    }
    metric_labels = {
        "error_among_accepted": "Error/accepted",
        "wrong_accept_per_task": "Wrong accept",
        "correct_completion_rate": "Correct completion",
        "mean_cost": "Cost",
    }
    deltas = result["coupled_active"]["paired_difference_stability"]
    lines = [
        r"\begin{tabular}{@{}lllll@{}}",
        r"\toprule",
        r"Baseline & Error/accepted & Wrong accept & Correct completion & Cost\\",
        r"\midrule",
    ]
    for baseline, label in labels.items():
        cells = []
        for metric in metric_labels:
            cell = deltas[baseline][metric]
            avg = cell["mean"]
            lo = cell["bootstrap_lo"]
            hi = cell["bootstrap_hi"]
            cells.append(f"{avg:+.4f} [{lo:+.4f},{hi:+.4f}]")
        lines.append(f"{label} & " + " & ".join(cells) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    COUPLED_DELTA_TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_coupled_stability_latex(result: dict[str, object]) -> None:
    labels = {
        "bayesian_depth2": "Bayesian plug-in depth 2",
        "static_worst_depth2": "Static worst-model depth 2",
        "fixed_broad": "Fixed broad check",
    }
    metrics = {
        "error_among_accepted": "Error/accepted",
        "wrong_accept_per_task": "Wrong accept",
        "correct_completion_rate": "Correct completion",
        "mean_cost": "Cost",
    }
    stability = result["coupled_active"]["paired_difference_stability"]
    lines = [
        r"\begin{tabular}{@{}lllll@{}}",
        r"\toprule",
        r"Baseline & Metric & Central 95\% range & Improves / ties / hurts & Mean\\",
        r"\midrule",
    ]
    for baseline, baseline_label in labels.items():
        for metric, metric_label in metrics.items():
            cell = stability[baseline][metric]
            lines.append(
                "{} & {} & [{:+.4f},{:+.4f}] & {:.2f}/{:.2f}/{:.2f} & {:+.4f}\\\\".format(
                    baseline_label,
                    metric_label,
                    cell["central_lo"],
                    cell["central_hi"],
                    cell["improved"],
                    cell["tied"],
                    cell["worsened"],
                    cell["mean"],
                )
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    COUPLED_STABILITY_TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_coupled_disagreement_latex(result: dict[str, object]) -> None:
    labels = {
        ("bayesian_depth2", "low_risk"): "Bayes, lower-risk",
        ("static_worst_depth2", "low_risk"): "Static, lower-risk",
        ("bayesian_depth2", "all"): "Bayes, all histories",
        ("static_worst_depth2", "all"): "Static, all histories",
    }
    attribution = result["coupled_active"]["disagreement_attribution"]
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Comparison & Diff. route & Extra triage & Avoided wrong & Correct diff & Cost diff & Width\\",
        r"\midrule",
    ]
    for (baseline, history), label in labels.items():
        row = attribution[baseline][history]
        lines.append(
            "{} & {:.3f} & {:.3f} & {:+.3f} & {:+.3f} & {:+.3f} & {:.3f}\\\\".format(
                label,
                row["different_rate"],
                row["dynamic_extra_triage"],
                row["avoided_wrong_accept"],
                -row["lost_correct_accept"],
                row["cost_delta"],
                row["wilson_width"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    COUPLED_DISAGREEMENT_TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_shift_latex(result: dict[str, object]) -> None:
    labels = {
        "independent_bayes": "Independent Bayes",
        "correlated_plugin": "Correlated plug-in",
        "bayesian_upper": "Bayesian upper",
        "credal_dependence": "Credal upper",
        "dedup_then_orthogonal": "Dedup + orthogonal",
        "always_orthogonal": "Always orthogonal",
        "abstain_all": "Abstain all",
    }
    summary = result["shift_stress"]["summary"]
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Policy & Accept & Error/accepted & Wrong accept & Correct completion & Cost\\",
        r"\midrule",
    ]
    for policy in DEPENDENCE_POLICIES:
        row = summary[policy]
        lines.append(
            "{} & {} & {} & {} & {} & {}\\\\".format(
                labels[policy],
                fmt(row["accept_rate"]),
                fmt(row["error_among_accepted"]),
                fmt(row["wrong_accept_per_task"]),
                fmt(row["correct_completion_rate"]),
                fmt(row["mean_cost"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    SHIFT_TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_necessity_latex(result: dict[str, object]) -> None:
    labels = {
        "dynamic_credal": "Dynamic credal",
        "bayesian_history_plugin": "History plug-in Bayes",
        "scalar_same_risk": "Scalar same-risk point",
        "static_interface": "Static interface endpoint",
        "static_boundary": "Static boundary endpoint",
        "fixed_broad": "Fixed broad check",
        "full_check": "Always full check",
    }
    summary = result["necessity_stress"]["summary"]
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Policy & Accept & Error/accepted & Wrong accept & Correct completion & Cost & Targeted\\",
        r"\midrule",
    ]
    for policy, label in labels.items():
        row = summary[policy]
        lines.append(
            "{} & {} & {} & {} & {} & {} & {}\\\\".format(
                label,
                fmt(row["accept_rate"]),
                fmt(row["error_among_accepted"]),
                fmt(row["wrong_accept_per_task"]),
                fmt(row["correct_completion_rate"]),
                fmt(row["mean_cost"]),
                fmt(row["targeted_query_rate"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    NECESSITY_TEX_OUT.write_text("\n".join(lines), encoding="utf-8")

    paths = result["necessity_stress"]["first_query_by_history"]
    history_labels = {
        "interface_context": "Interface context",
        "boundary_context": "Boundary context",
        "ambiguous_context": "Ambiguous context",
    }
    path_lines = [
        r"\begin{tabular}{@{}llll@{}}",
        r"\toprule",
        r"History & Dynamic credal & Scalar same-risk & Fixed broad\\",
        r"\midrule",
    ]
    for history_name, label in history_labels.items():
        path_lines.append(
            "{} & {} & {} & {}\\\\".format(
                label,
                _dominant_query_name(paths["dynamic_credal"][history_name]),
                _dominant_query_name(paths["scalar_same_risk"][history_name]),
                _dominant_query_name(paths["fixed_broad"][history_name]),
            )
        )
    path_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    NECESSITY_PATH_TEX_OUT.write_text("\n".join(path_lines), encoding="utf-8")

    trace_rows = []
    for history_name, query_name in (
        ("interface_context", "interface_check"),
        ("boundary_context", "boundary_check"),
        ("ambiguous_context", "broad_check"),
    ):
        beliefs = NECESSITY_CREDAL_BELIEFS[history_name]
        initial = max(error_risk(belief) for belief in beliefs)
        for observation in NECESSITY_QUERIES[query_name].observation_likelihood:
            updated = update_belief_set(beliefs, query_name, observation, NECESSITY_QUERIES)
            upper = max(error_risk(belief) for belief in updated)
            trace_rows.append((history_name, query_name, observation, initial, upper, upper <= DELTA))
    trace_labels = {
        "interface_context": "Interface context",
        "boundary_context": "Boundary context",
        "ambiguous_context": "Ambiguous context",
        "interface_check": "interface",
        "boundary_check": "boundary",
        "broad_check": "broad",
    }
    trace_lines = [
        r"\begin{tabular}{@{}lllrrl@{}}",
        r"\toprule",
        r"History & Check & Observation & Initial upper risk & Updated upper risk & Terminal action\\",
        r"\midrule",
    ]
    for history_name, query_name, observation, initial, upper, accept in trace_rows:
        action = "accept" if accept else "decline"
        trace_lines.append(
            "{} & {} & {} & {} & {} & {}\\\\".format(
                trace_labels[history_name],
                trace_labels[query_name],
                observation.replace("_", r"\_"),
                fmt(initial),
                fmt(upper),
                action,
            )
        )
    trace_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    NECESSITY_TRACE_TEX_OUT.write_text("\n".join(trace_lines), encoding="utf-8")


def _dominant_query_name(first_query: dict[str, float]) -> str:
    label, value = max(first_query.items(), key=lambda item: item[1])
    names = {
        "interface_check": "interface",
        "boundary_check": "boundary",
        "broad_check": "broad",
        "full_check": "full",
        "none": "none",
    }
    return f"{names[label]} ({value:.2f})"


def run(
    *,
    seed: int = SEED,
    calibration_reps: int = CALIBRATION_REPS,
    test_tasks_per_rep: int = TEST_TASKS_PER_REP,
    dynamic_tasks: int = DYNAMIC_TASKS,
    necessity_tasks: int = NECESSITY_TASKS,
    shift_tasks: int = 20_000,
) -> dict[str, object]:
    rng = random.Random(seed)
    result = {
        "protocol": {
            "version": "verifier-dependence-pilot-v2",
            "seed": seed,
            "delta": DELTA,
            "decline_loss": DECLINE_LOSS,
            "cheap_check_cost": CHEAP_CHECK_COST,
            "orthogonal_check_cost": ORTHOGONAL_CHECK_COST,
            "true_pass": TRUE_PASS,
            "false_pass_marginal": FALSE_PASS_MARGINAL,
        },
        "same_distribution": run_dependence_same_distribution(
            rng,
            calibration_reps=calibration_reps,
            test_tasks_per_rep=test_tasks_per_rep,
        ),
        "dynamic_checks": run_dynamic_checks(rng, tasks=dynamic_tasks),
        "coupled_active": run_coupled_active_experiment(
            rng,
            calibration_reps=calibration_reps,
            test_tasks_per_rep=test_tasks_per_rep,
        ),
        "necessity_stress": run_credal_necessity_stress(
            rng,
            tasks=necessity_tasks,
        ),
        "shift_stress": run_shift_stress(rng, tasks=shift_tasks),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--calibration-reps", type=int, default=CALIBRATION_REPS)
    parser.add_argument("--test-tasks-per-rep", type=int, default=TEST_TASKS_PER_REP)
    parser.add_argument("--dynamic-tasks", type=int, default=DYNAMIC_TASKS)
    parser.add_argument("--necessity-tasks", type=int, default=NECESSITY_TASKS)
    parser.add_argument("--shift-tasks", type=int, default=20_000)
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = run(
        seed=args.seed,
        calibration_reps=args.calibration_reps,
        test_tasks_per_rep=args.test_tasks_per_rep,
        dynamic_tasks=args.dynamic_tasks,
        necessity_tasks=args.necessity_tasks,
        shift_tasks=args.shift_tasks,
    )
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_dependence_latex(result)
    write_dynamic_latex(result)
    write_coupled_latex(result)
    write_coupled_path_latex(result)
    write_coupled_delta_latex(result)
    write_coupled_stability_latex(result)
    write_coupled_disagreement_latex(result)
    write_shift_latex(result)
    write_necessity_latex(result)
    printable = {
        "protocol": result["protocol"],
        "same_distribution": result["same_distribution"],
        "dynamic_checks": result["dynamic_checks"],
        "coupled_active": {
            key: value for key, value in result["coupled_active"].items()
            if key != "per_rep_summary"
        },
        "necessity_stress": result["necessity_stress"],
        "shift_stress": {
            key: value for key, value in result["shift_stress"].items()
            if key != "summary"
        },
    }
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
