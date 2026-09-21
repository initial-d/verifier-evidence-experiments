#!/usr/bin/env python3
"""Prior-mismatch stress replay on public-history hidden checks.

This analysis does not turn the development-selected public fixtures into a
natural-distribution benchmark.  It asks a narrower question: if a deployment
point prior is highly confident that the simple visible branch is sufficient,
does the same two-branch purchase threshold seen in the controlled studies
also separate policies when the final labels are real public-repository hidden
checks?
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "experiments" / "results"
MINED = RESULT_DIR / "mined_issue_fixture.json"
CROSS = RESULT_DIR / "cross_repo_source_pattern_fixture.json"
JSON_OUT = RESULT_DIR / "public_history_prior_stress.json"
TEX_OUT = RESULT_DIR / "public_history_prior_stress_table.tex"
APPENDIX_TEX_OUT = RESULT_DIR / "public_history_prior_stress_appendix_table.tex"
SOURCE_TRANSFER_TEX_OUT = RESULT_DIR / "public_history_source_transfer_table.tex"
CROSS_REPO_TEX_OUT = RESULT_DIR / "public_history_cross_repo_stress_table.tex"
CONFIDENCE_TRANSFER_TEX_OUT = RESULT_DIR / "public_history_confidence_transfer_table.tex"


POINT_MISSED_BRANCH = 0.05
CAUTIOUS_POINT_MISSED_BRANCH = 0.10
CREDAL_RHO = 0.75
WILSON_Z_ONE_SIDED_95 = 1.645


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _fraction(rate: float, total: int) -> str:
    return f"{round(rate * total)}/{total}"


def _buys(missed_branch_mass: float, incremental_cost: float) -> bool:
    return missed_branch_mass > incremental_cost + 1e-12


def _wilson_upper(successes: int, total: int, z: float = WILSON_Z_ONE_SIDED_95) -> float:
    if total <= 0:
        return 1.0
    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = p_hat + (z * z) / (2.0 * total)
    radius = z * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z * z) / (4.0 * total * total))
    return min(1.0, (center + radius) / denom)


def _collect_records() -> list[dict[str, object]]:
    mined = json.loads(MINED.read_text(encoding="utf-8"))
    cross = json.loads(CROSS.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    def add_suite(data: dict[str, object], simple_policy: str, source_policy: str, suite: str) -> None:
        by_task: dict[str, dict[str, dict[str, object]]] = {}
        for row in data["records"]:
            task_id = str(row["task_id"])
            by_task.setdefault(task_id, {})[str(row["policy"])] = row
        for task_id, policies in sorted(by_task.items()):
            simple = policies[simple_policy]
            source = policies[source_policy]
            rows.append({
                "suite": suite,
                "task_id": task_id,
                "issue_id": simple.get("issue_id") or simple.get("fixture_id"),
                "repo": simple.get("repo", "pypa/packaging"),
                "simple_success": float(simple["success"]),
                "source_success": float(source["success"]),
                "simple_cost": float(simple["cost"]),
                "source_cost": float(source["cost"]),
                "rescued": float(source["success"]) - float(simple["success"]),
            })

    add_suite(mined, "E_patch_coverage", "F_source_pattern_generated", "packaging")
    add_suite(cross, "E_naive_shallow_pattern", "F_source_pattern_generated", "cross_repo")
    return rows


def run() -> dict[str, object]:
    records = _collect_records()
    total = len(records)
    simple_success = mean(row["simple_success"] for row in records)
    source_success = mean(row["source_success"] for row in records)
    simple_cost = mean(row["simple_cost"] for row in records)
    source_cost = mean(row["source_cost"] for row in records)
    incremental_cost = source_cost - simple_cost
    credal_effective_miss = (1.0 - CREDAL_RHO) * POINT_MISSED_BRANCH + CREDAL_RHO
    aggregate_point_buys = _buys(POINT_MISSED_BRANCH, incremental_cost)
    aggregate_cautious_point_buys = _buys(CAUTIOUS_POINT_MISSED_BRANCH, incremental_cost)
    aggregate_credal_buys = _buys(credal_effective_miss, incremental_cost)

    policies = [
        {
            "policy": "E_visible_simple",
            "description": "strongest simple visible comparator",
            "success_rate": simple_success,
            "mean_cost": simple_cost,
            "utility_gap": source_success - simple_success,
            "query_rate": 0.0,
        },
        {
            "policy": "B_high_confidence_point",
            "description": "same-interface point prior with 5% missed-branch mass",
            "success_rate": source_success if aggregate_point_buys else simple_success,
            "mean_cost": source_cost if aggregate_point_buys else simple_cost,
            "utility_gap": 0.0 if aggregate_point_buys else source_success - simple_success,
            "query_rate": 1.0 if aggregate_point_buys else 0.0,
        },
        {
            "policy": "D_credal_stress",
            "description": "epsilon-contamination credal stress replay",
            "success_rate": source_success if aggregate_credal_buys else simple_success,
            "mean_cost": source_cost if aggregate_credal_buys else simple_cost,
            "utility_gap": 0.0 if aggregate_credal_buys else source_success - simple_success,
            "query_rate": 1.0 if aggregate_credal_buys else 0.0,
        },
        {
            "policy": "G_cautious_point",
            "description": "cautious point prior with 10% missed-branch mass",
            "success_rate": source_success if aggregate_cautious_point_buys else simple_success,
            "mean_cost": source_cost if aggregate_cautious_point_buys else simple_cost,
            "utility_gap": 0.0 if aggregate_cautious_point_buys else source_success - simple_success,
            "query_rate": 1.0 if aggregate_cautious_point_buys else 0.0,
        },
    ]
    transfer = []
    confidence_transfer = []
    for train_suite, test_suite in (("packaging", "cross_repo"), ("cross_repo", "packaging")):
        train = [row for row in records if row["suite"] == train_suite]
        test = [row for row in records if row["suite"] == test_suite]
        train_missed = mean(row["rescued"] for row in train)
        train_rescues = int(round(sum(float(row["rescued"]) for row in train)))
        train_incremental_cost = mean(row["source_cost"] - row["simple_cost"] for row in train)
        test_simple_success = mean(row["simple_success"] for row in test)
        test_source_success = mean(row["source_success"] for row in test)
        test_simple_cost = mean(row["simple_cost"] for row in test)
        test_source_cost = mean(row["source_cost"] for row in test)
        transfer_credal_mass = (1.0 - CREDAL_RHO) * train_missed + CREDAL_RHO
        finite_sample_upper = _wilson_upper(train_rescues, len(train))
        transfer_point_buys = _buys(train_missed, train_incremental_cost)
        transfer_credal_buys = _buys(transfer_credal_mass, train_incremental_cost)
        finite_sample_credal_buys = _buys(finite_sample_upper, train_incremental_cost)
        transfer.append({
            "train_suite": train_suite,
            "test_suite": test_suite,
            "train_checks": len(train),
            "train_rescues": train_rescues,
            "test_checks": len(test),
            "train_missed_branch": train_missed,
            "train_incremental_cost": train_incremental_cost,
            "credal_effective_missed_branch": transfer_credal_mass,
            "finite_sample_upper_branch": finite_sample_upper,
            "point_buys": transfer_point_buys,
            "credal_buys": transfer_credal_buys,
            "finite_sample_credal_buys": finite_sample_credal_buys,
            "point_success_rate": test_source_success if transfer_point_buys else test_simple_success,
            "credal_success_rate": test_source_success if transfer_credal_buys else test_simple_success,
            "finite_sample_credal_success_rate": test_source_success if finite_sample_credal_buys else test_simple_success,
            "point_mean_cost": test_source_cost if transfer_point_buys else test_simple_cost,
            "credal_mean_cost": test_source_cost if transfer_credal_buys else test_simple_cost,
            "finite_sample_credal_mean_cost": test_source_cost if finite_sample_credal_buys else test_simple_cost,
            "test_simple_success_rate": test_simple_success,
            "test_source_success_rate": test_source_success,
        })
        confidence_transfer.append({
            "train_suite": train_suite,
            "test_suite": test_suite,
            "train_rescues": train_rescues,
            "train_checks": len(train),
            "plugin_mass": train_missed,
            "wilson_upper": finite_sample_upper,
            "incremental_cost": train_incremental_cost,
            "plugin_buys": transfer_point_buys,
            "credal_buys": finite_sample_credal_buys,
            "plugin_success_rate": test_source_success if transfer_point_buys else test_simple_success,
            "credal_success_rate": test_source_success if finite_sample_credal_buys else test_simple_success,
            "credal_gain": (
                (test_source_success if finite_sample_credal_buys else test_simple_success)
                - (test_source_success if transfer_point_buys else test_simple_success)
            ),
        })
    leave_one_repo = []
    for repo in sorted({row["repo"] for row in records}):
        train = [row for row in records if row["repo"] != repo]
        test = [row for row in records if row["repo"] == repo]
        train_missed = mean(row["rescued"] for row in train)
        train_incremental_cost = mean(row["source_cost"] - row["simple_cost"] for row in train)
        test_simple_success = mean(row["simple_success"] for row in test)
        test_source_success = mean(row["source_success"] for row in test)
        test_simple_cost = mean(row["simple_cost"] for row in test)
        test_source_cost = mean(row["source_cost"] for row in test)
        credal_mass = (1.0 - CREDAL_RHO) * train_missed + CREDAL_RHO
        point_buys = _buys(train_missed, train_incremental_cost)
        credal_buys = _buys(credal_mass, train_incremental_cost)
        leave_one_repo.append({
            "heldout_repo": repo,
            "train_checks": len(train),
            "test_checks": len(test),
            "train_missed_branch": train_missed,
            "train_incremental_cost": train_incremental_cost,
            "point_buys": point_buys,
            "credal_buys": credal_buys,
            "point_success_rate": test_source_success if point_buys else test_simple_success,
            "credal_success_rate": test_source_success if credal_buys else test_simple_success,
            "point_mean_cost": test_source_cost if point_buys else test_simple_cost,
            "credal_mean_cost": test_source_cost if credal_buys else test_simple_cost,
            "test_simple_success_rate": test_simple_success,
            "test_source_success_rate": test_source_success,
        })
    source_to_rest = []
    for repo in sorted({row["repo"] for row in records}):
        train = [row for row in records if row["repo"] == repo]
        test = [row for row in records if row["repo"] != repo]
        train_missed = mean(row["rescued"] for row in train)
        train_incremental_cost = mean(row["source_cost"] - row["simple_cost"] for row in train)
        test_simple_success = mean(row["simple_success"] for row in test)
        test_source_success = mean(row["source_success"] for row in test)
        test_simple_cost = mean(row["simple_cost"] for row in test)
        test_source_cost = mean(row["source_cost"] for row in test)
        credal_mass = (1.0 - CREDAL_RHO) * train_missed + CREDAL_RHO
        point_buys = _buys(train_missed, train_incremental_cost)
        credal_buys = _buys(credal_mass, train_incremental_cost)
        source_to_rest.append({
            "train_repo": repo,
            "train_checks": len(train),
            "test_checks": len(test),
            "train_missed_branch": train_missed,
            "train_incremental_cost": train_incremental_cost,
            "credal_effective_missed_branch": credal_mass,
            "point_buys": point_buys,
            "credal_buys": credal_buys,
            "point_success_rate": test_source_success if point_buys else test_simple_success,
            "credal_success_rate": test_source_success if credal_buys else test_simple_success,
            "point_mean_cost": test_source_cost if point_buys else test_simple_cost,
            "credal_mean_cost": test_source_cost if credal_buys else test_simple_cost,
            "test_simple_success_rate": test_simple_success,
            "test_source_success_rate": test_source_success,
        })
    result = {
        "suite": "public-history prior-mismatch stress replay",
        "scope": (
            "development-selected public-history hidden checks; tests threshold behavior "
            "under an explicit high-confidence point prior, not natural-distribution performance"
        ),
        "checks": total,
        "issues": len({row["issue_id"] for row in records}),
        "repos": len({row["repo"] for row in records}),
        "simple_successes": int(round(simple_success * total)),
        "source_successes": int(round(source_success * total)),
        "rescued_checks": int(round((source_success - simple_success) * total)),
        "simple_cost": simple_cost,
        "source_cost": source_cost,
        "incremental_query_cost": incremental_cost,
        "point_missed_branch": POINT_MISSED_BRANCH,
        "cautious_point_missed_branch": CAUTIOUS_POINT_MISSED_BRANCH,
        "credal_rho": CREDAL_RHO,
        "credal_effective_missed_branch": credal_effective_miss,
        "point_buys": aggregate_point_buys,
        "cautious_point_buys": aggregate_cautious_point_buys,
        "credal_buys": aggregate_credal_buys,
        "policies": policies,
        "slice_transfer": transfer,
        "finite_sample_confidence_transfer": confidence_transfer,
        "leave_one_repo_transfer": leave_one_repo,
        "source_to_rest_transfer": source_to_rest,
        "cross_repo_stress": _cross_repo_stress(records),
        "records": records,
    }
    return result


def _cross_repo_stress(records: list[dict[str, object]]) -> dict[str, object]:
    train = [row for row in records if row["suite"] == "packaging"]
    rows = [row for row in records if row["suite"] == "cross_repo"]
    train_missed = mean(row["rescued"] for row in train)
    train_incremental_cost = mean(row["source_cost"] - row["simple_cost"] for row in train)
    simple_success = mean(row["simple_success"] for row in rows)
    source_success = mean(row["source_success"] for row in rows)
    simple_cost = mean(row["simple_cost"] for row in rows)
    source_cost = mean(row["source_cost"] for row in rows)
    credal_effective_miss = (1.0 - CREDAL_RHO) * train_missed + CREDAL_RHO
    point_buys = _buys(train_missed, train_incremental_cost)
    cautious_point_buys = _buys(CAUTIOUS_POINT_MISSED_BRANCH, train_incremental_cost)
    credal_buys = _buys(credal_effective_miss, train_incremental_cost)
    policies = [
        {
            "policy": "E_visible_simple",
            "success_rate": simple_success,
            "mean_cost": simple_cost,
            "query_rate": 0.0,
        },
        {
            "policy": "B_high_confidence_point",
            "success_rate": source_success if point_buys else simple_success,
            "mean_cost": source_cost if point_buys else simple_cost,
            "query_rate": 1.0 if point_buys else 0.0,
        },
        {
            "policy": "D_credal_stress",
            "success_rate": source_success if credal_buys else simple_success,
            "mean_cost": source_cost if credal_buys else simple_cost,
            "query_rate": 1.0 if credal_buys else 0.0,
        },
        {
            "policy": "G_cautious_point",
            "success_rate": source_success if cautious_point_buys else simple_success,
            "mean_cost": source_cost if cautious_point_buys else simple_cost,
            "query_rate": 1.0 if cautious_point_buys else 0.0,
        },
    ]
    return {
        "checks": len(rows),
        "repos": len({row["repo"] for row in rows}),
        "simple_successes": int(round(simple_success * len(rows))),
        "source_successes": int(round(source_success * len(rows))),
        "train_checks": len(train),
        "train_missed_branch": train_missed,
        "train_incremental_cost": train_incremental_cost,
        "test_source_cost": source_cost,
        "test_simple_cost": simple_cost,
        "point_missed_branch": train_missed,
        "cautious_point_missed_branch": CAUTIOUS_POINT_MISSED_BRANCH,
        "credal_effective_missed_branch": credal_effective_miss,
        "point_buys": point_buys,
        "cautious_point_buys": cautious_point_buys,
        "credal_buys": credal_buys,
        "policies": policies,
    }


def write_latex(result: dict[str, object]) -> None:
    labels = {
        "E_visible_simple": "Visible simple comparator",
        "B_high_confidence_point": "High-confidence point prior",
        "D_credal_stress": "Credal stress replay",
        "G_cautious_point": "Cautious point prior",
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Public-history prior-mismatch stress replay. The hidden labels are the real public-history checks from Table~\ref{tab:public-history-summary}, but the branch prior is a stress setting rather than an estimate of the natural issue distribution. The high-confidence point prior assigns 5\% mass to the complete source-pattern branch; the observed mean incremental source-pattern cost is 0.079.}",
        r"\label{tab:public-history-prior-stress}",
        r"\small",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Policy & Success & Cost & Utility gap & Query rate \\",
        r"\midrule",
    ]
    for row in result["policies"]:
        lines.append(
            f"{labels[row['policy']]} & {_fmt(float(row['success_rate']))} & "
            f"{_fmt(float(row['mean_cost']))} & {_fmt(float(row['utility_gap']))} & "
            f"{_fmt(float(row['query_rate']))} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        (
            f"% Public-history stress: simple={_fraction(float(result['simple_successes']) / int(result['checks']), int(result['checks']))}; "
            f"source={_fraction(float(result['source_successes']) / int(result['checks']), int(result['checks']))}; "
            f"rescued={result['rescued_checks']}/{result['checks']}; "
            f"incremental_cost={_fmt(float(result['incremental_query_cost']))}; "
            f"credal_effective_missed_branch={_fmt(float(result['credal_effective_missed_branch']))}"
        ),
    ])
    def yes_no(value: bool) -> str:
        return "yes" if value else "no"

    lines.extend([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Slice-to-slice same-interface prior-transfer replay on public-history checks. The point prior and purchase threshold are estimated only from the training slice; the reported success is measured on the held-out slice.}",
        r"\label{tab:public-history-prior-transfer}",
        r"\small",
        r"\begin{adjustbox}{width=\linewidth}",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Train & Test & Missed mass & Inc. cost & Point buys & Credal buys & Point success & Credal gain \\",
        r"\midrule",
    ])
    for row in result["slice_transfer"]:
        gain = float(row["credal_success_rate"]) - float(row["point_success_rate"])
        lines.append(
            f"{row['train_suite'].replace('_', ' ')} & {row['test_suite'].replace('_', ' ')} & "
            f"{_fmt(float(row['train_missed_branch']))} & {_fmt(float(row['train_incremental_cost']))} & "
            f"{yes_no(bool(row['point_buys']))} & {yes_no(bool(row['credal_buys']))} & "
            f"{_fmt(float(row['point_success_rate']))} & {_fmt(gain)} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
        "",
    ])
    appendix_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Leave-one-repository prior-transfer robustness check. Each row estimates the point missed-branch mass and purchase threshold on all other repositories, then evaluates on the held-out repository. Most held-out rows are ties because the cross-repository training set is already cautious enough; the packaging-to-other-repository slice in Table~\ref{tab:public-history-prior-transfer} remains the separated public-history stress case.}",
        r"\label{tab:public-history-leave-one-repo}",
        r"\small",
        r"\begin{adjustbox}{width=\linewidth}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Held-out repository & Missed mass & Inc. cost & Point success & Point cost & Credal success & Credal cost \\",
        r"\midrule",
    ]
    for row in result["leave_one_repo_transfer"]:
        appendix_lines.append(
            f"{row['heldout_repo']} & {_fmt(float(row['train_missed_branch']))} & "
            f"{_fmt(float(row['train_incremental_cost']))} & {_fmt(float(row['point_success_rate']))} & "
            f"{_fmt(float(row['point_mean_cost']))} & {_fmt(float(row['credal_success_rate']))} & "
            f"{_fmt(float(row['credal_mean_cost']))} \\\\"
        )
    appendix_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
        "",
    ])
    source_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Single-source-repository prior-transfer grid. Each row estimates the point missed-branch mass and incremental source-pattern cost from one repository, then evaluates on all other repositories.}",
        r"\label{tab:public-history-source-transfer}",
        r"\small",
        r"\begin{adjustbox}{width=\linewidth}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Train repository & Train checks & Test checks & Missed mass & Inc. cost & Point buys & Point success & Credal gain \\",
        r"\midrule",
    ]
    for row in result["source_to_rest_transfer"]:
        gain = float(row["credal_success_rate"]) - float(row["point_success_rate"])
        repo = str(row["train_repo"]).replace("_", r"\_")
        source_lines.append(
            f"{repo} & {row['train_checks']} & {row['test_checks']} & "
            f"{_fmt(float(row['train_missed_branch']))} & {_fmt(float(row['train_incremental_cost']))} & "
            f"{yes_no(bool(row['point_buys']))} & {_fmt(float(row['point_success_rate']))} & {_fmt(gain)} \\\\"
        )
    source_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
        "",
    ])
    TEX_OUT.write_text("\n".join(lines), encoding="utf-8")
    APPENDIX_TEX_OUT.write_text("\n".join(appendix_lines), encoding="utf-8")
    SOURCE_TRANSFER_TEX_OUT.write_text("\n".join(source_lines), encoding="utf-8")
    confidence_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Finite-sample credal prior-transfer replay. The plug-in point policy uses the empirical rescued-branch mass from the training slice. The confidence-set credal policy uses the one-sided 95\% Wilson upper envelope from the same training labels. Both policies are then evaluated on the held-out slice without retuning. This table is a small-sample stress diagnostic, not a natural-distribution benchmark.}",
        r"\label{tab:public-history-confidence-transfer}",
        r"\small",
        r"\begin{adjustbox}{width=\linewidth}",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"Train & Test & Rescues & Plug-in mass & Wilson upper & Inc. cost & Plug-in buys & Credal buys & Credal gain \\",
        r"\midrule",
    ]
    for row in result["finite_sample_confidence_transfer"]:
        confidence_lines.append(
            f"{row['train_suite'].replace('_', ' ')} & {row['test_suite'].replace('_', ' ')} & "
            f"{row['train_rescues']}/{row['train_checks']} & "
            f"{_fmt(float(row['plugin_mass']))} & {_fmt(float(row['wilson_upper']))} & "
            f"{_fmt(float(row['incremental_cost']))} & {yes_no(bool(row['plugin_buys']))} & "
            f"{yes_no(bool(row['credal_buys']))} & {_fmt(float(row['credal_gain']))} \\\\"
        )
    confidence_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
        "",
    ])
    CONFIDENCE_TRANSFER_TEX_OUT.write_text("\n".join(confidence_lines), encoding="utf-8")
    cross = result["cross_repo_stress"]
    cross_labels = {
        "E_visible_simple": "Visible simple comparator",
        "B_high_confidence_point": "High-confidence point prior",
        "D_credal_stress": "Credal stress replay",
        "G_cautious_point": "Cautious point prior",
    }
    cross_lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Cross-repository held-out prior-transfer replay. The point missed-branch mass and purchase threshold are estimated on the \texttt{{packaging}} slice, then applied without retuning to the {cross['checks']} hidden checks from four non-\texttt{{packaging}} repositories.}}",
        r"\label{tab:public-history-cross-repo-stress}",
        r"\small",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Policy & Success & Test cost & Query rate & Train mass/cost & Buys source \\",
        r"\midrule",
    ]
    for row in cross["policies"]:
        cross_lines.append(
            f"{cross_labels[row['policy']]} & {_fmt(float(row['success_rate']))} & "
            f"{_fmt(float(row['mean_cost']))} & {_fmt(float(row['query_rate']))} & "
            f"{_fmt(float(cross['train_missed_branch']))}/{_fmt(float(cross['train_incremental_cost']))} & "
            f"{yes_no(float(row['query_rate']) > 0.0)} \\\\"
        )
    cross_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        (
            f"% Cross-repo stress: simple={cross['simple_successes']}/{cross['checks']}; "
            f"source={cross['source_successes']}/{cross['checks']}; "
            f"train_incremental_cost={_fmt(float(cross['train_incremental_cost']))}; "
            f"point_buys={yes_no(bool(cross['point_buys']))}; "
            f"credal_buys={yes_no(bool(cross['credal_buys']))}"
        ),
    ])
    CROSS_REPO_TEX_OUT.write_text("\n".join(cross_lines), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = run()
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_latex(result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
