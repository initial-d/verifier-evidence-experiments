"""Smoke checks for the public artifact bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_text(name: str) -> str:
    return (RESULTS / name).read_text(encoding="utf-8")


def manifest_hash(records: list[dict]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    require(
        prospective["manifest"]["manifest_sha256"] == manifest_hash(prospective["manifest"]["records"]),
        "prospective benchmark manifest hash changed",
    )

    locked = load("locked_public_history_replay.json")
    require(locked["issues"] == 12, "locked public-history issue count changed")
    require(
        locked["manifest_sha256"] == manifest_hash(locked["records"]),
        "locked public-history manifest hash changed",
    )
    locked_table = load_text("locked_public_history_replay_table.tex")
    require(
        locked["manifest_sha256"][:12] in locked_table,
        "locked public-history table does not include the manifest hash prefix",
    )

    print("artifact smoke check passed")


if __name__ == "__main__":
    main()
