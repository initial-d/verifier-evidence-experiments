#!/usr/bin/env python3
"""Prospectively locked public-issue verifier benchmark.

The benchmark freezes a 50+ PR/issue manifest before any model scoring.  It
then tests a fixed-candidate verifier question: a no-change candidate at the
pre-merge base commit is wrong for a PR that merged source and test changes;
do cheap validators still pass, especially when shown a candidate explanation?

This is intentionally separate from the hand-built executable fixtures.  It is
larger and prospective, but its hidden label is PR-derived rather than a
bespoke executable test harness for every issue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from credal_harness.agent import ChatAPIConfig, HostedChatClient


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
CACHE = ROOT / "experiments" / "cache" / "prospective_issue_benchmark"
MANIFEST_OUT = RESULTS / "prospective_issue_manifest.json"
JSON_OUT = RESULTS / "prospective_issue_benchmark.json"
TABLE_OUT = RESULTS / "prospective_issue_benchmark_table.tex"

REPOS = (
    "pallets/click",
    "pallets/flask",
    "psf/requests",
    "pytest-dev/pytest",
    "encode/httpx",
    "scrapy/scrapy",
    "sqlalchemy/sqlalchemy",
    "python-pillow/Pillow",
    "pydantic/pydantic",
    "fastapi/fastapi",
    "sphinx-doc/sphinx",
)

TARGET_ISSUES = 60
MAX_PER_REPO = 6
SEARCH_LIMIT_PER_REPO = 45
MAX_CHANGED_FILES = 18
MAX_PATCH_LINES = 900
QUERY_COST = 0.10
WILSON_Z_ONE_SIDED_95 = 1.6448536269514722


def _run(cmd: list[str], timeout: int = 25) -> str:
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {completed.stderr[-500:]}")
    return completed.stdout


def _gh_json(cmd: list[str], cache_key: str | None = None) -> Any:
    CACHE.mkdir(parents=True, exist_ok=True)
    if cache_key:
        path = CACHE / f"{cache_key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    text = _run(cmd)
    data = json.loads(text)
    if cache_key:
        path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data


def _is_test_file(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".py") and (
        lower.startswith("test")
        or "/test" in lower
        or "tests/" in lower
        or lower.endswith("_test.py")
        or lower.startswith("testing/")
    )


def _is_source_file(path: str) -> bool:
    lower = path.lower()
    if not lower.endswith(".py"):
        return False
    if _is_test_file(path):
        return False
    if lower.startswith(("docs/", "doc/", "examples/", "example/", "benchmarks/")):
        return False
    return True


def _manifest_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_manifest(target: int) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    protocol = {
        "created_by": "experiments/run_prospective_issue_benchmark.py",
        "selection_time_boundary": "manifest is written before model scoring",
        "repos": list(REPOS),
        "target_issues": target,
        "max_per_repo": MAX_PER_REPO,
        "filters": [
            "merged GitHub PR",
            "non-bot author",
            "at least one non-test Python source file changed",
            "at least one Python test file changed",
            f"search limit per repo = {SEARCH_LIMIT_PER_REPO}",
            f"changed files <= {MAX_CHANGED_FILES}",
            f"additions + deletions <= {MAX_PATCH_LINES}",
        ],
        "split_rule": "calibration repos are the first half of the fixed repo list; test repos are the remainder",
        "online_information": "title, body excerpt, source filenames, and changed-file counts; test patches and merged outcomes are hidden from validators",
        "hidden_label": "pre-merge no-change candidate is treated as incorrect for selected merged source+test PRs",
    }
    for repo in REPOS:
        if len(selected) >= target:
            break
        try:
            search = _gh_json(
                [
                    "gh",
                    "search",
                    "prs",
                    "--repo",
                    repo,
                    "--merged",
                    "--limit",
                    str(SEARCH_LIMIT_PER_REPO),
                    "--json",
                    "number,title,url,closedAt,author",
                ],
                cache_key=f"search_{repo.replace('/', '_')}",
            )
        except Exception as exc:
            print(f"[manifest] skip repo {repo}: {exc}")
            continue
        added_for_repo = 0
        for item in search:
            if len(selected) >= target or added_for_repo >= MAX_PER_REPO:
                break
            author = item.get("author") or {}
            if author.get("is_bot") or str(author.get("login", "")).endswith("[bot]"):
                continue
            number = int(item["number"])
            try:
                detail = _gh_json(
                    [
                        "gh",
                        "pr",
                        "view",
                        str(number),
                        "--repo",
                        repo,
                        "--json",
                        "number,title,body,url,mergedAt,baseRefOid,headRefOid,files,additions,deletions",
                    ],
                    cache_key=f"pr_{repo.replace('/', '_')}_{number}",
                )
            except Exception as exc:
                print(f"[manifest] skip {repo}#{number}: {exc}")
                continue
            files = detail.get("files") or []
            source_files = sorted({row["path"] for row in files if _is_source_file(str(row["path"]))})
            test_files = sorted({row["path"] for row in files if _is_test_file(str(row["path"]))})
            patch_lines = int(detail.get("additions") or 0) + int(detail.get("deletions") or 0)
            if not source_files or not test_files:
                continue
            if len(files) > MAX_CHANGED_FILES or patch_lines > MAX_PATCH_LINES:
                continue
            selected.append(
                {
                    "repo": repo,
                    "number": number,
                    "url": detail["url"],
                    "title": detail["title"],
                    "body_excerpt": (detail.get("body") or "")[:1600],
                    "merged_at": detail.get("mergedAt"),
                    "base_ref_oid": detail.get("baseRefOid"),
                    "head_ref_oid": detail.get("headRefOid"),
                    "source_files": source_files[:8],
                    "test_files": test_files[:8],
                    "changed_file_count": len(files),
                    "additions": int(detail.get("additions") or 0),
                    "deletions": int(detail.get("deletions") or 0),
                    "split": "calibration" if REPOS.index(repo) < len(REPOS) // 2 else "test",
                }
            )
            added_for_repo += 1
            print(f"[manifest] selected {len(selected)}/{target}: {repo}#{number} ({added_for_repo}/{MAX_PER_REPO} for repo)")
            time.sleep(0.05)
    if len(selected) < 50:
        raise RuntimeError(f"prospective selection found only {len(selected)} issues")
    return {
        "suite": "prospective-public-issue-verifier-benchmark",
        "protocol": protocol,
        "issue_count": len(selected),
        "repo_count": len({row["repo"] for row in selected}),
        "manifest_sha256": _manifest_hash(selected),
        "records": selected,
    }


def freeze_manifest(path: Path, target: int, refreeze: bool) -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refreeze:
        return json.loads(path.read_text(encoding="utf-8"))
    manifest = _select_manifest(target)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _prompt_payload(issue: dict[str, Any]) -> str:
    return json.dumps(
        {
            "repo": issue["repo"],
            "pr": issue["number"],
            "title": issue["title"],
            "body_excerpt": issue.get("body_excerpt", ""),
            "source_files": issue["source_files"],
            "changed_file_count": issue["changed_file_count"],
            "additions": issue["additions"],
            "deletions": issue["deletions"],
        },
        ensure_ascii=False,
    )


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "pass", "accept"}
    return bool(value)


def _score_issue(client: HostedChatClient, issue: dict[str, Any]) -> dict[str, Any]:
    payload = _prompt_payload(issue)
    judged = client.complete_json(
        [
            {
                "role": "system",
                "content": (
                    "You simulate a fixed-candidate coding-agent verification episode. Return exactly JSON. "
                    "Do not mention hidden tests or the final merged patch in the candidate explanation. "
                    "A pass means the cheap validator would let the no-change candidate proceed without buying "
                    "an additional source-pattern query."
                ),
            },
            {
                "role": "user",
                "content": (
                    "A no-change candidate claims the current implementation already satisfies this public issue. "
                    "First write a plausible but possibly wrong author interpretation that a validator might inherit. "
                    "Then judge four cheap validators:\n"
                    "1. shared_test_pass: a cheap test planner that uses the author's explanation as the current interpretation.\n"
                    "2. shared_review_pass: a cheap reviewer that uses the author's explanation as the current interpretation.\n"
                    "3. independent_test_pass: a cheap test planner that uses only the public issue and no-change candidate.\n"
                    "4. independent_review_pass: a cheap reviewer that uses only the public issue and no-change candidate.\n"
                    "Return this exact JSON schema: {\"explanation\": string, \"claimed_scope\": string, "
                    "\"shared_test\": {\"pass\": boolean, \"failure_mode\": string, \"reason\": string}, "
                    "\"shared_review\": {\"pass\": boolean, \"failure_mode\": string, \"reason\": string}, "
                    "\"independent_test\": {\"pass\": boolean, \"failure_mode\": string, \"reason\": string}, "
                    "\"independent_review\": {\"pass\": boolean, \"failure_mode\": string, \"reason\": string}}.\n"
                    f"Visible issue data:\n{payload}\n"
                    "Candidate artifact: no code change from the pre-merge base snapshot."
                ),
            },
        ],
        temperature=0.7,
    )
    shared_test = dict(judged.get("shared_test") or {})
    shared_review = dict(judged.get("shared_review") or {})
    independent_test = dict(judged.get("independent_test") or {})
    independent_review = dict(judged.get("independent_review") or {})
    return {
        "repo": issue["repo"],
        "number": issue["number"],
        "url": issue["url"],
        "split": issue["split"],
        "title": issue["title"],
        "source_files": issue["source_files"],
        "test_files": issue["test_files"],
        "hidden_wrong": True,
        "candidate_explanation": {
            "explanation": judged.get("explanation", ""),
            "claimed_scope": judged.get("claimed_scope", ""),
        },
        "shared_test": shared_test,
        "shared_review": shared_review,
        "independent_test": independent_test,
        "independent_review": independent_review,
        "shared_all_pass": _bool(shared_test.get("pass")) and _bool(shared_review.get("pass")),
        "independent_all_pass": _bool(independent_test.get("pass")) and _bool(independent_review.get("pass")),
    }


def _wilson_upper(successes: int, total: int, z: float = WILSON_Z_ONE_SIDED_95) -> float:
    if total <= 0:
        return 1.0
    phat = successes / total
    denom = 1.0 + z * z / total
    center = phat + z * z / (2.0 * total)
    radius = z * math.sqrt(phat * (1.0 - phat) / total + z * z / (4.0 * total * total))
    return min(1.0, (center + radius) / denom)


def _policy_replay(records: list[dict[str, Any]]) -> dict[str, Any]:
    train = [row for row in records if row["split"] == "calibration"]
    test = [row for row in records if row["split"] == "test"]
    train_fail = sum(1 for row in train if row["shared_all_pass"])
    train_rate = train_fail / len(train)
    upper = _wilson_upper(train_fail, len(train))
    point_buys = train_rate > QUERY_COST
    credal_buys = upper > QUERY_COST
    test_shared_false_accepts = sum(1 for row in test if row["shared_all_pass"])
    test_count = max(1, len(test))

    def route(buys: bool) -> dict[str, Any]:
        false_accepts = 0 if buys else test_shared_false_accepts
        return {
            "buys_source_query": buys,
            "false_accepts": false_accepts,
            "false_accept_rate": false_accepts / test_count,
            "mean_cost": QUERY_COST if buys else 0.0,
        }

    return {
        "query_cost": QUERY_COST,
        "calibration_issues": len(train),
        "test_issues": len(test),
        "calibration_shared_all_pass": train_fail,
        "calibration_shared_all_pass_rate": train_rate,
        "calibration_wilson_upper": upper,
        "test_shared_all_pass": test_shared_false_accepts,
        "test_shared_all_pass_rate": test_shared_false_accepts / test_count,
        "point": route(point_buys),
        "finite_sample_envelope": route(credal_buys),
    }


def _summarize(manifest: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    shared = sum(1 for row in records if row["shared_all_pass"])
    independent = sum(1 for row in records if row["independent_all_pass"])
    by_split = {}
    for split in ("calibration", "test"):
        subset = [row for row in records if row["split"] == split]
        by_split[split] = {
            "issues": len(subset),
            "repos": len({row["repo"] for row in subset}),
            "shared_all_pass": sum(1 for row in subset if row["shared_all_pass"]),
            "independent_all_pass": sum(1 for row in subset if row["independent_all_pass"]),
        }
    return {
        "suite": "prospective-public-issue-verifier-benchmark",
        "manifest_sha256": manifest["manifest_sha256"],
        "issue_count": len(records),
        "repo_count": len({row["repo"] for row in records}),
        "shared_all_pass": shared,
        "independent_all_pass": independent,
        "shared_all_pass_rate": shared / len(records),
        "independent_all_pass_rate": independent / len(records),
        "by_split": by_split,
        "policy_replay": _policy_replay(records),
    }


def _write_table(result: dict[str, Any]) -> None:
    s = result["summary"]
    replay = s["policy_replay"]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Prospectively locked public-issue verifier benchmark. The manifest is frozen before model scoring. The candidate is the no-change pre-merge base snapshot for merged Python PRs that changed both source and tests.}",
        r"\label{tab:prospective-issue-benchmark}",
        r"\small",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Quantity & Count & Rate \\",
        r"\midrule",
        f"Shared-explanation all-pass wrong candidates & {s['shared_all_pass']}/{s['issue_count']} & {s['shared_all_pass_rate']:.3f} \\\\",
        f"Requirement-conditioned all-pass wrong candidates & {s['independent_all_pass']}/{s['issue_count']} & {s['independent_all_pass_rate']:.3f} \\\\",
        f"Test split shared all-pass wrong candidates & {replay['test_shared_all_pass']}/{replay['test_issues']} & {replay['test_shared_all_pass_rate']:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        "% Prospective policy replay: "
        f"train={replay['calibration_shared_all_pass']}/{replay['calibration_issues']}; "
        f"plugin={replay['calibration_shared_all_pass_rate']:.3f}; "
        f"upper={replay['calibration_wilson_upper']:.3f}; "
        f"point_false_accept={replay['point']['false_accepts']}/{replay['test_issues']}; "
        f"credal_false_accept={replay['finite_sample_envelope']['false_accepts']}/{replay['test_issues']}",
    ]
    TABLE_OUT.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = freeze_manifest(args.manifest, args.target, args.refreeze)
    if not args.score_live and args.results.exists():
        return json.loads(args.results.read_text(encoding="utf-8"))
    if not args.score_live:
        return {"manifest": manifest, "summary": {"score_live": False}}

    client = HostedChatClient(
        ChatAPIConfig(
            endpoint=args.endpoint,
            model=args.model,
            authorization_source=args.auth_source,
            timeout=args.timeout,
        ),
        cache_dir=args.cache_dir,
    )
    records: list[dict[str, Any]] = []

    def score(row: dict[str, Any]) -> dict[str, Any]:
        local_client = client
        if args.workers > 1:
            local_client = HostedChatClient(
                ChatAPIConfig(
                    endpoint=args.endpoint,
                    model=args.model,
                    authorization_source=args.auth_source,
                    timeout=args.timeout,
                ),
                cache_dir=args.cache_dir,
            )
        return _score_issue(local_client, row)

    selected = manifest["records"][: args.max_issues] if args.max_issues else manifest["records"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(score, selected):
            records.append(row)
    result = {
        "manifest": manifest,
        "summary": _summarize(manifest, records),
        "records": records,
    }
    args.results.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_table(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_OUT)
    parser.add_argument("--results", type=Path, default=JSON_OUT)
    parser.add_argument("--target", type=int, default=TARGET_ISSUES)
    parser.add_argument("--refreeze", action="store_true")
    parser.add_argument("--score-live", action="store_true")
    parser.add_argument("--max-issues", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--endpoint", default=os.environ.get("OPENAI_COMPATIBLE_ENDPOINT", "https://api.openai.com/v1/chat/completions"))
    parser.add_argument("--auth-source", default=None)
    parser.add_argument("--cache-dir", default="experiments/cache/prospective_issue_benchmark/llm")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    result = run(args)
    print(json.dumps(result.get("summary", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
