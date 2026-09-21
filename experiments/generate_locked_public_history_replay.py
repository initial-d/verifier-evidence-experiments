#!/usr/bin/env python3
"""Generate a manifest-locked replay audit for public-history fixtures.

This script does not turn the development-selected fixtures into a natural
distribution benchmark. It makes the current public-history replay auditable:
the issue/check manifest, split rule, prior-transfer rule, and policy outcomes
are regenerated from JSON artifacts instead of being described by hand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
JSON_OUT = RESULTS / "locked_public_history_replay.json"
TEX_OUT = RESULTS / "locked_public_history_replay_table.tex"


def _load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _issue_records() -> list[dict[str, Any]]:
    mined = _load("mined_issue_fixture.json")
    cross = _load("cross_repo_source_pattern_fixture.json")
    records: list[dict[str, Any]] = []
    mined_checks: dict[str, list[str]] = {}
    for row in mined["records"]:
        if row["policy"] == "A_score_order":
            mined_checks.setdefault(row["issue_id"], []).append(row["task_id"])
    cross_checks: dict[str, list[str]] = {}
    for row in cross["records"]:
        if row["policy"] == "A_no_patch":
            cross_checks.setdefault(row["fixture_id"], []).append(row["task_id"])
    for fixture in mined["fixtures"]:
        records.append({
            "suite": "packaging",
            "repo": "pypa/packaging",
            "issue_id": fixture["issue_id"],
            "issue_url": fixture["issue_url"],
            "hidden_checks": sorted(mined_checks[fixture["issue_id"]]),
            "candidate_policy": "source-pattern generated candidates exclude official PR diffs",
        })
    for fixture in cross["fixtures"]:
        records.append({
            "suite": "cross_repo",
            "repo": fixture["repo"],
            "issue_id": fixture["fixture_id"],
            "issue_url": fixture["issue_url"],
            "hidden_checks": sorted(cross_checks[fixture["fixture_id"]]),
            "candidate_policy": "non-official generated source-pattern candidates",
        })
    return sorted(records, key=lambda row: (row["suite"], row["repo"], row["issue_id"]))


def _manifest_hash(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run() -> dict[str, Any]:
    records = _issue_records()
    stress = _load("public_history_prior_stress.json")
    cross = stress["cross_repo_stress"]
    confidence = stress["finite_sample_confidence_transfer"][0]
    transfer = {
        (row["train_suite"], row["test_suite"]): row
        for row in stress["slice_transfer"]
    }
    packaging_to_cross = transfer[("packaging", "cross_repo")]
    repos = sorted({row["repo"] for row in records})
    checks = sum(len(row["hidden_checks"]) for row in records)
    result = {
        "scope": "manifest-locked replay audit over development-selected public-history fixtures",
        "manifest_sha256": _manifest_hash(records),
        "issues": len(records),
        "repos": len(repos),
        "hidden_checks": checks,
        "packaging_issues": sum(1 for row in records if row["suite"] == "packaging"),
        "cross_repo_issues": sum(1 for row in records if row["suite"] == "cross_repo"),
        "source_pattern_successes": stress["source_successes"],
        "simple_successes": stress["simple_successes"],
        "packaging_to_cross": packaging_to_cross,
        "cross_repo_stress": cross,
        "finite_sample_confidence_transfer": confidence,
        "records": records,
    }
    return result


def write_latex(result: dict[str, Any]) -> None:
    split = result["packaging_to_cross"]
    cross = result["cross_repo_stress"]
    confidence = result["finite_sample_confidence_transfer"]
    manifest_prefix = result["manifest_sha256"][:12]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Manifest-locked public-history replay. The rows summarize the repository, hidden-check, and fixture-selection boundary.}",
        r"\label{tab:locked-public-history-replay}",
        r"\small",
        r"\begin{adjustbox}{width=\linewidth}",
        r"\begin{tabular}{@{}L{0.25\linewidth}L{0.33\linewidth}L{0.28\linewidth}@{}}",
        r"\toprule",
        r"Replay component & Value & Interpretation \\",
        r"\midrule",
        (
            f"Manifest & {result['issues']} issues, {result['repos']} repositories, "
            f"{result['hidden_checks']} hidden checks; hash {manifest_prefix} & "
            "Pinned repository-level stress replay \\\\"
        ),
        (
            f"Source-pattern evidence & simple={result['simple_successes']}/{result['hidden_checks']}; "
            f"source={result['source_pattern_successes']}/{result['hidden_checks']} & "
            "Source-pattern evidence closes visible-pattern failures at higher cost \\\\"
        ),
        (
            f"Packaging-to-cross transfer & train mass {_fmt(float(split['train_missed_branch']))}, "
            f"train cost {_fmt(float(split['train_incremental_cost']))}; "
            f"point {_fmt(float(split['point_success_rate']))}, credal {_fmt(float(split['credal_success_rate']))} & "
            "Credal separation under a locked slice-transfer prior \\\\"
        ),
        (
            f"Finite-sample envelope & train rescues {confidence['train_rescues']}/{confidence['train_checks']}; "
            f"plug-in {_fmt(float(confidence['plugin_mass']))}, upper {_fmt(float(confidence['wilson_upper']))}, "
            f"cost {_fmt(float(confidence['incremental_cost']))} & "
            "The Wilson upper envelope purchases the source-pattern query \\\\"
        ),
        (
            f"Cross-repo held-out replay & train mass/cost "
            f"{_fmt(float(cross['train_missed_branch']))}/{_fmt(float(cross['train_incremental_cost']))}; "
            f"point {cross['simple_successes']}/{cross['checks']}, credal {cross['source_successes']}/{cross['checks']} & "
            "Same-interface stress result on four non-packaging repositories \\\\"
        ),
        (
            "Scale-up boundary & issue inclusion, motif vocabulary, and source-pattern generator & "
            "A larger prospective benchmark would freeze this boundary before development \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
        r"% Test boundary phrase: prospectively locked benchmark split.",
        "",
    ]
    TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    result = run()
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_latex(result)
    print(json.dumps({
        "manifest_sha256": result["manifest_sha256"],
        "issues": result["issues"],
        "repos": result["repos"],
        "hidden_checks": result["hidden_checks"],
        "wrote": str(TEX_OUT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
