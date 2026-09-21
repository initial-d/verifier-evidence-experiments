"""Smoke checks for the public artifact bundle."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    require(not any(ROOT.glob("*.pdf")), "paper PDF should not be committed at repository root")
    require(not any(ROOT.glob("*submission*.tex")), "manuscript source should not be committed")

    verifier = load("verifier_dependence_pilot.json")
    require("same_distribution" in verifier, "controlled dependence result is missing")
    require("necessity_stress" in verifier, "same-risk result is missing")

    shared = load("shared_misunderstanding_real_summary.json")
    shared_config = shared["config"]
    require(
        shared_config["calibration_n_per_condition"] + shared_config["test_candidates_per_condition_per_rep"] == 360,
        "fixed-candidate replay size changed",
    )

    hosted = load("llm_shared_misunderstanding_qwen_turbo_family_split.json")
    require(hosted["num_splits"] > 0, "hosted-model split replay is missing")

    prospective = load("prospective_issue_benchmark.json")
    require(prospective["summary"]["issue_count"] == 59, "prospective benchmark issue count changed")

    locked = load("locked_public_history_replay.json")
    require(locked["issues"] == 12, "locked public-history issue count changed")

    print("artifact smoke check passed")


if __name__ == "__main__":
    main()
