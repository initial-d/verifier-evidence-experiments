#!/usr/bin/env python3
"""Hosted-LLM shared-misunderstanding replay.

The experiment keeps the fixed-candidate boundary from
``run_shared_misunderstanding_real.py`` but lets a hosted model naturally
produce the implementation, explanation, generated-test plan, and review.
Live calls are opt-in; the default offline mode is only a smoke test for the
replay machinery.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping

from credal_harness.agent import ChatAPIConfig, HostedChatClient


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OUT = RESULTS / "llm_shared_misunderstanding.json"
TABLE_OUT = RESULTS / "llm_shared_misunderstanding_table.tex"
CASE_OUT = RESULTS / "llm_shared_misunderstanding_cases_table.tex"
FAMILY_SPLIT_OUT = RESULTS / "llm_shared_misunderstanding_qwen_turbo_family_split.json"
FAMILY_SPLIT_TABLE_OUT = RESULTS / "llm_shared_misunderstanding_qwen_turbo_family_split_table.tex"

SEED = 20260918
DELTA = 0.05
DECLINE_LOSS = 0.16
PROBE_COST = 0.035
Z_ONE_SIDED_95 = 1.6448536269514722
POINT_THRESHOLD_GRID = tuple(index / 1000.0 for index in range(0, int(DELTA * 1000) + 1))
ATOM_SPACE = (
    ("correct", True, True),
    ("mutation", False, True),
    ("edge", True, False),
    ("both", False, False),
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    family: str
    requirement: str
    visible_input: Any
    probe_inputs: tuple[Any, ...]
    hidden_inputs: tuple[Any, ...]
    reference: Callable[[Any], Any]
    offline: Mapping[str, str] | None = None


@dataclass(frozen=True)
class Candidate:
    task: TaskSpec
    code: str
    explanation: str
    live_generated: bool


def _normalize_tags(data: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in data:
        tag = item.strip().lower()
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _active_names(data: list[dict[str, Any]]) -> list[str]:
    return [row["name"].strip().lower() for row in data if row.get("active", False)]


def _available_stock(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"sku": row["sku"], "available": row["stock"] - row.get("reserved", 0)}
        for row in data
        if row["stock"] - row.get("reserved", 0) > 0
    ]


def _merge_defaults(data: dict[str, Any]) -> dict[str, Any]:
    return {"timeout": 30, "retries": 3, **data}


def _sort_numbers(data: list[int]) -> list[int]:
    return sorted(data)


def _append_checksum(data: list[int]) -> list[int]:
    return list(data) + [sum(data)]


TASK_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "family": "normalize_tags",
        "reference": _normalize_tags,
        "requirement": (
            "Implement solve(data). Return a new list of tags after stripping whitespace, "
            "lowercasing, removing empty strings, and removing duplicates while preserving "
            "first occurrence order. Do not modify the input list."
        ),
        "visible": [" Red ", "Blue"],
        "probes": (["Red", "red"], ["", " A "]),
        "hidden": ([" A ", "B", "a"], ["", " C ", "c"]),
        "offline": {
            "correct": "def solve(data):\n    seen=set(); out=[]\n    for item in data:\n        tag=item.strip().lower()\n        if tag and tag not in seen:\n            seen.add(tag); out.append(tag)\n    return out\n",
            "mutation": "def solve(data):\n    for i,item in enumerate(data):\n        data[i]=item.strip().lower()\n    return [x for x in data if x]\n",
            "edge": "def solve(data):\n    return [item.strip().lower() for item in data if item.strip()]\n",
        },
    },
    {
        "family": "active_names",
        "reference": _active_names,
        "requirement": (
            "Implement solve(data). Return a new list containing lowercase stripped names "
            "only for records whose active field is true. Do not mutate the input records."
        ),
        "visible": [{"name": " Ana ", "active": True}, {"name": "Bo", "active": True}],
        "probes": (
            [{"name": " Ana ", "active": True}, {"name": "Cid", "active": False}],
        ),
        "hidden": (
            [{"name": " DEE ", "active": True}],
            [{"name": "Eve", "active": False}, {"name": "Fox", "active": True}],
        ),
        "offline": {
            "correct": "def solve(data):\n    return [row['name'].strip().lower() for row in data if row.get('active', False)]\n",
            "mutation": "def solve(data):\n    out=[]\n    for row in data:\n        row['name']=row['name'].strip().lower()\n        if row.get('active', False): out.append(row['name'])\n    return out\n",
            "edge": "def solve(data):\n    return [row['name'].strip().lower() for row in data]\n",
        },
    },
    {
        "family": "available_stock",
        "reference": _available_stock,
        "requirement": (
            "Implement solve(data). Return a new list of dictionaries with sku and available "
            "quantity equal to stock minus reserved, omitting items whose available quantity "
            "is not positive. Missing reserved means zero. Do not mutate input rows."
        ),
        "visible": [{"sku": "a", "stock": 3, "reserved": 1}],
        "probes": ([{"sku": "a", "stock": 1, "reserved": 1}, {"sku": "b", "stock": 5}],),
        "hidden": (
            [{"sku": "c", "stock": 2, "reserved": 5}],
            [{"sku": "d", "stock": 0}, {"sku": "e", "stock": 2}],
        ),
        "offline": {
            "correct": "def solve(data):\n    return [{'sku': row['sku'], 'available': row['stock']-row.get('reserved',0)} for row in data if row['stock']-row.get('reserved',0)>0]\n",
            "mutation": "def solve(data):\n    out=[]\n    for row in data:\n        row['available']=row['stock']-row.get('reserved',0)\n        if row['available']>0: out.append({'sku': row['sku'], 'available': row['available']})\n    return out\n",
            "edge": "def solve(data):\n    return [{'sku': row['sku'], 'available': row['stock']-row.get('reserved',0)} for row in data]\n",
        },
    },
    {
        "family": "merge_defaults",
        "reference": _merge_defaults,
        "requirement": (
            "Implement solve(data). Return a new dictionary containing default timeout=30 "
            "and retries=3, overridden by any supplied keys even when the supplied value is "
            "zero or false. Do not modify the input dictionary."
        ),
        "visible": {"timeout": 10},
        "probes": ({"timeout": 0},),
        "hidden": ({"retries": 0}, {}, {"timeout": 5, "retries": 0}),
        "offline": {
            "correct": "def solve(data):\n    return {'timeout': 30, 'retries': 3, **data}\n",
            "mutation": "def solve(data):\n    data.setdefault('timeout',30); data.setdefault('retries',3); return data\n",
            "edge": "def solve(data):\n    out={'timeout':30,'retries':3}\n    for k,v in data.items():\n        if v: out[k]=v\n    return out\n",
        },
    },
    {
        "family": "sort_numbers",
        "reference": _sort_numbers,
        "requirement": (
            "Implement solve(data). Return a new sorted list of the input numbers, preserving "
            "duplicates and negative values. Do not modify the input list."
        ),
        "visible": [3, 1, 2],
        "probes": ([2, 2, 1],),
        "hidden": ([3, 1, 2, 1], [], [-2, 5, -2]),
        "offline": {
            "correct": "def solve(data):\n    return sorted(data)\n",
            "mutation": "def solve(data):\n    data.sort(); return data\n",
            "edge": "def solve(data):\n    return sorted(set(data))\n",
        },
    },
    {
        "family": "append_checksum",
        "reference": _append_checksum,
        "requirement": (
            "Implement solve(data). Return a new list equal to the input values followed by "
            "their arithmetic sum. Preserve duplicates and negative values. Do not modify the "
            "input list."
        ),
        "visible": [4, 5],
        "probes": ([2, 2],),
        "hidden": ([4, 5, 4], [], [-1, 6]),
        "offline": {
            "correct": "def solve(data):\n    return list(data)+[sum(data)]\n",
            "mutation": "def solve(data):\n    data.append(sum(data)); return data\n",
            "edge": "def solve(data):\n    return list(data)+[sum(set(data))]\n",
        },
    },
)


def _parameterized_task(index: int) -> TaskSpec:
    choice = index % 12
    suffix = index // 12
    if choice == 0:
        def reference(data: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for item in data:
                tag = item.strip().lower()
                if tag and tag not in seen:
                    seen.add(tag)
                    out.append(tag)
            return out

        visible = [f" Red {suffix}", "Blue"]
        probes = ([f"A{suffix}", f" a{suffix} ", ""],)
        hidden = ([f" C{suffix} ", f"c{suffix}", ""], ["", f" Z{suffix} "])
        return TaskSpec(
            task_id=f"llm_shared_{index:04d}",
            family="normalize_tags",
            requirement=(
                "Implement solve(data). Return a new list of tags after stripping whitespace, "
                "lowercasing, removing empty strings, and removing duplicates while preserving "
                "first occurrence order. Do not modify the input list."
            ),
            visible_input=visible,
            probe_inputs=probes,
            hidden_inputs=hidden,
            reference=reference,
        )
    if choice == 1:
        flag = "active" if suffix % 2 == 0 else "enabled"
        field = "name" if suffix % 2 == 0 else "title"

        def reference(data: list[dict[str, Any]]) -> list[str]:
            return [row[field].strip().lower() for row in data if row.get(flag, False)]

        visible = [{field: f" Ana {suffix} ", flag: True}, {field: "Bo", flag: True}]
        probes = ([{field: "Cid", flag: False}, {field: f" Dee {suffix} ", flag: True}],)
        hidden = ([{field: "Eve", flag: False}, {field: "Fox", flag: True}],)
        return TaskSpec(
            task_id=f"llm_shared_{index:04d}",
            family="filter_records",
            requirement=(
                f"Implement solve(data). Return a new list containing lowercase stripped {field} "
                f"values only for records whose {flag} field is true. Missing {flag} means false. "
                "Do not mutate the input records."
            ),
            visible_input=visible,
            probe_inputs=probes,
            hidden_inputs=hidden,
            reference=reference,
        )
    if choice == 2:
        stock_key = "stock" if suffix % 2 == 0 else "quantity"
        reserved_key = "reserved" if suffix % 2 == 0 else "held"

        def reference(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {"sku": row["sku"], "available": row[stock_key] - row.get(reserved_key, 0)}
                for row in data
                if row[stock_key] - row.get(reserved_key, 0) > 0
            ]

        visible = [{"sku": f"a{suffix}", stock_key: 3, reserved_key: 1}]
        probes = ([{"sku": "zero", stock_key: 1, reserved_key: 1}, {"sku": "miss", stock_key: 5}],)
        hidden = ([{"sku": "neg", stock_key: 2, reserved_key: 5}, {"sku": "ok", stock_key: 2}],)
        return TaskSpec(
            task_id=f"llm_shared_{index:04d}",
            family="available_stock",
            requirement=(
                f"Implement solve(data). Return a new list of dictionaries with sku and available "
                f"quantity equal to {stock_key} minus {reserved_key}, omitting rows whose available "
                f"quantity is not positive. Missing {reserved_key} means zero. Do not mutate input rows."
            ),
            visible_input=visible,
            probe_inputs=probes,
            hidden_inputs=hidden,
            reference=reference,
        )
    if choice == 3:
        defaults = {"timeout": 30 + suffix % 5, "retries": 3, "enabled": True}

        def reference(data: dict[str, Any]) -> dict[str, Any]:
            return {**defaults, **data}

        visible = {"timeout": 10 + suffix}
        probes = ({"timeout": 0, "enabled": False},)
        hidden = ({"retries": 0}, {}, {"timeout": 5, "enabled": False})
        return TaskSpec(
            task_id=f"llm_shared_{index:04d}",
            family="merge_defaults",
            requirement=(
                "Implement solve(data). Return a new dictionary containing defaults "
                f"{json.dumps(defaults, sort_keys=True)}, overridden by supplied keys even when "
                "the supplied value is zero or false. Do not modify the input dictionary."
            ),
            visible_input=visible,
            probe_inputs=probes,
            hidden_inputs=hidden,
            reference=reference,
        )
    if choice == 4:
        def reference(data: list[int]) -> list[int]:
            return sorted(data, key=lambda value: (abs(value), 0 if value < 0 else 1))

        visible = [3 + suffix, 1, 2]
        probes = ([2, -2, 1],)
        hidden = ([-1, 1, 0, 2, -2], [], [-3 - suffix, 3 + suffix, 1])
        return TaskSpec(
            task_id=f"llm_shared_{index:04d}",
            family="sort_numbers",
            requirement=(
                "Implement solve(data). Return a new list sorted by absolute value. When two numbers "
                "have the same absolute value, put the negative number before the positive number. "
                "Preserve duplicates. Do not modify the input list."
            ),
            visible_input=visible,
            probe_inputs=probes,
            hidden_inputs=hidden,
            reference=reference,
        )
    if choice == 5:
        def reference(data: list[int]) -> list[int]:
            checksum = sum((pos + 1) * value for pos, value in enumerate(data))
            return list(data) + [checksum]

        visible = [4 + suffix, 5]
        probes = ([2, 2, -1],)
        hidden = ([4, 5, 4], [], [-1, 6, suffix])
        return TaskSpec(
            task_id=f"llm_shared_{index:04d}",
            family="append_checksum",
            requirement=(
                "Implement solve(data). Return a new list equal to the input values followed by "
                "a weighted checksum, where each input value is multiplied by its one-based position "
                "before summing. Preserve duplicates and negative values. Do not modify the input list."
            ),
            visible_input=visible,
            probe_inputs=probes,
            hidden_inputs=hidden,
            reference=reference,
        )
    if choice == 6:
        def reference(data: list[str]) -> dict[str, str]:
            out: dict[str, str] = {}
            for item in data:
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                key = key.strip().lower()
                if key:
                    out[key] = value.strip()
            return out

        visible = [f"mode = fast{suffix}", "path=/tmp"]
        probes = (["bad", "mode = slow", "mode = final"],)
        hidden = (["=skip", "flag=false", "count=0"], ["x=1=2"])
        return TaskSpec(
            task_id=f"llm_shared_{index:04d}",
            family="parse_assignments",
            requirement=(
                "Implement solve(data). Return a new dictionary parsed from strings of the form key=value. "
                "Strip whitespace, lowercase keys, keep values as strings, skip malformed rows and empty keys, "
                "and let later duplicates override earlier ones. Do not modify the input list."
            ),
            visible_input=visible,
            probe_inputs=probes,
            hidden_inputs=hidden,
            reference=reference,
        )
    if choice == 7:
        def reference(data: list[dict[str, Any]]) -> dict[str, int]:
            out: dict[str, int] = {}
            for row in data:
                if "id" not in row:
                    continue
                value = int(row.get("score", 0))
                out[str(row["id"])] = max(0, min(100, value))
            return out

        visible = [{"id": f"a{suffix}", "score": 80}]
        probes = ([{"id": "low", "score": -3}, {"id": "high", "score": 120}, {"id": "miss"}],)
        hidden = ([{"id": "z", "score": 0}, {"score": 5}], [{"id": "h", "score": 101}])
        return TaskSpec(
            task_id=f"llm_shared_{index:04d}",
            family="clamp_scores",
            requirement=(
                "Implement solve(data). Return a new dictionary mapping each row id to an integer score "
                "clamped into the inclusive range 0..100. Missing score means zero, and rows without id are skipped. "
                "Do not mutate the input records."
            ),
            visible_input=visible,
            probe_inputs=probes,
            hidden_inputs=hidden,
            reference=reference,
        )
    if choice == 8:
        def reference(data: list[int]) -> list[int]:
            return [data[i] + data[i + 1] for i in range(0, len(data) - 1, 2)]

        visible = [1 + suffix, 2, 3, 4]
        probes = ([5], [1, 2, 3])
        hidden = ([], [4], [-1, 2, -3, 4, 99])
        return TaskSpec(
            task_id=f"llm_shared_{index:04d}",
            family="adjacent_sums",
            requirement=(
                "Implement solve(data). Return a new list containing sums of non-overlapping adjacent pairs "
                "at positions (0,1), (2,3), and so on. Drop a final unpaired value. For lists with fewer than "
                "two elements, return an empty list. Do not modify the input list."
            ),
            visible_input=visible,
            probe_inputs=probes,
            hidden_inputs=hidden,
            reference=reference,
        )
    if choice == 9:
        def reference(data: dict[str, Any]) -> list[dict[str, Any]]:
            reserved = data.get("reserved", {})
            out = []
            for row in data.get("items", []):
                available = row["count"] - reserved.get(row["sku"], 0)
                if available > 0:
                    out.append({"sku": row["sku"], "available": available})
            return sorted(out, key=lambda row: row["sku"])

        visible = {"items": [{"sku": f"a{suffix}", "count": 3}], "reserved": {}}
        probes = ({"items": [{"sku": "b", "count": 1}, {"sku": "a", "count": 3}], "reserved": {"b": 1}},)
        hidden = ({"items": [{"sku": "c", "count": 2}], "reserved": {"c": 3}}, {"items": []})
        return TaskSpec(
            task_id=f"llm_shared_{index:04d}",
            family="nested_inventory",
            requirement=(
                "Implement solve(data). Input is a dictionary with items and an optional reserved dictionary mapping "
                "sku strings to reserved counts. Return a new list of sku/available dictionaries where available is "
                "item count minus reserved for that sku, omit nonpositive availability, and sort the returned list by sku. "
                "Do not modify the input dictionary or item records."
            ),
            visible_input=visible,
            probe_inputs=probes,
            hidden_inputs=hidden,
            reference=reference,
        )
    if choice == 10:
        def reference(data: list[Any]) -> Any:
            for item in data:
                if item is not None and item != "":
                    return item
            return None

        visible = ["", f"value{suffix}"]
        probes = ([None, "", 0, "later"], [None, "", False])
        hidden = ([None, "", []], [None, "x"], [])
        return TaskSpec(
            task_id=f"llm_shared_{index:04d}",
            family="first_present",
            requirement=(
                "Implement solve(data). Return the first element that is not None and not the empty string. "
                "Values such as 0, False, and [] count as present and must be returned. If none exists, return None. "
                "Do not modify the input list."
            ),
            visible_input=visible,
            probe_inputs=probes,
            hidden_inputs=hidden,
            reference=reference,
        )
    def reference(data: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        out: list[dict[str, Any]] = []
        for group in data:
            for row in group:
                key = str(row["id"]).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append({"id": key, "value": row.get("value", 0)})
        return out

    visible = [[{"id": f"A{suffix}", "value": 1}], [{"id": "B", "value": 2}]]
    probes = ([[{"id": " A ", "value": 1}, {"id": "a", "value": 9}], []],)
    hidden = (([], [{"id": "", "value": 5}], [{"id": "C"}]), [[{"id": "x", "value": 1}], [{"id": " X ", "value": 2}]])
    return TaskSpec(
        task_id=f"llm_shared_{index:04d}",
        family="flatten_unique",
        requirement=(
            "Implement solve(data). Flatten a list of groups of records into a new list of dictionaries. "
            "Each output dictionary has a lowercase stripped id and the original value, defaulting missing value to zero. "
            "Remove later records whose normalized id was already seen, skip empty ids, and preserve first occurrence order. "
            "Do not modify the input nested lists or records."
        ),
        visible_input=visible,
        probe_inputs=probes,
        hidden_inputs=hidden,
        reference=reference,
    )


def build_tasks(n: int) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for index in range(n):
        task = _parameterized_task(index)
        tasks.append(TaskSpec(**{**task.__dict__, "requirement": f"Task {index + 1}. {task.requirement}"}))
    return tasks


def _safe_builtins() -> dict[str, Any]:
    return {
        "all": all,
        "abs": abs,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "Exception": Exception,
        "KeyError": KeyError,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "zip": zip,
    }


def _extract_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _load_solve(code: str) -> Callable[[Any], Any]:
    namespace: dict[str, Any] = {"__builtins__": _safe_builtins()}
    exec(_extract_code(code), namespace)
    solve = namespace.get("solve")
    if not callable(solve):
        raise ValueError("candidate did not define solve(data)")
    return solve


def _run_case(candidate: Candidate, input_value: Any) -> tuple[bool, bool]:
    before = copy.deepcopy(input_value)
    argument = copy.deepcopy(input_value)
    try:
        result = _load_solve(candidate.code)(argument)
    except Exception:
        return False, argument != before
    expected = candidate.task.reference(copy.deepcopy(input_value))
    return result == expected, argument != before


def _return_passes(candidate: Candidate, inputs: tuple[Any, ...]) -> bool:
    return all(_run_case(candidate, value)[0] for value in inputs)


def _mutation_passes(candidate: Candidate, inputs: tuple[Any, ...]) -> bool:
    return all(not _run_case(candidate, value)[1] for value in inputs)


def final_correct(candidate: Candidate) -> bool:
    return _return_passes(candidate, candidate.task.hidden_inputs) and _mutation_passes(
        candidate, candidate.task.hidden_inputs
    )


def final_state(candidate: Candidate) -> str:
    if final_correct(candidate):
        return "correct"
    mutation_fails = not _mutation_passes(candidate, candidate.task.hidden_inputs)
    return_fails = not _return_passes(candidate, candidate.task.hidden_inputs)
    if mutation_fails and return_fails:
        return "both"
    if mutation_fails:
        return "mutation"
    return "edge"


def mention_flags(text: str) -> dict[str, bool]:
    lowered = text.lower()
    mutation_terms = ("mutat", "in-place", "in place", "modify", "modifies", "original input", "side effect")
    edge_terms = ("duplicate", "empty", "zero", "false", "negative", "missing", "inactive", "edge")
    return {
        "mentions_mutation": any(term in lowered for term in mutation_terms),
        "mentions_edge": any(term in lowered for term in edge_terms),
    }


def visible_tag(
    condition: str,
    explanation: str,
    test_plan: Mapping[str, Any],
    review: Mapping[str, Any],
) -> str:
    shared_text = explanation if condition == "shared" else ""
    text = " ".join(
        [
            shared_text,
            json.dumps(test_plan, ensure_ascii=False),
            json.dumps(review, ensure_ascii=False),
        ]
    )
    flags = mention_flags(text)
    if not flags["mentions_mutation"] and not flags["mentions_edge"]:
        return "both_blind"
    if not flags["mentions_mutation"]:
        return "mutation_blind"
    if not flags["mentions_edge"]:
        return "edge_blind"
    return "both_visible"


def generate_candidate(
    client: HostedChatClient | None,
    task: TaskSpec,
    rng: random.Random,
    live: bool,
    temperature: float,
    quick_coder: bool,
) -> Candidate:
    if not live:
        family = next((row for row in TASK_FAMILIES if row["family"] == task.family), None)
        options = ("correct", "mutation", "edge")
        chosen = options[rng.randrange(len(options))]
        source = (
            family["offline"][chosen]
            if family is not None
            else "def solve(data):\n    return data\n"
        )
        return Candidate(
            task=task,
            code=source,
            explanation=(
                "I implement the visible transformation directly. "
                + ("I avoid modifying the input." if chosen != "mutation" else "")
            ),
            live_generated=False,
        )
    assert client is not None
    style = (
        "Write a compact first-pass implementation that handles the visible example and the main path. "
        "Do not include tests, and do not spend time on exhaustive edge-case analysis."
        if quick_coder
        else "Write a careful implementation."
    )
    response = client.complete_json(
        [
            {
                "role": "system",
                "content": (
                    "Return exactly JSON with keys code and explanation. The code must define "
                    "solve(data) and use only Python builtins. Do not include tests. "
                    + style
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Requirement:\n{task.requirement}\n\n"
                    f"Visible example input:\n{json.dumps(task.visible_input, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=temperature,
    )
    return Candidate(
        task=task,
        code=str(response.get("code", "")),
        explanation=str(response.get("explanation", "")),
        live_generated=True,
    )


def validator_test_plan(
    client: HostedChatClient | None,
    candidate: Candidate,
    condition: str,
    live: bool,
    temperature: float,
) -> Mapping[str, Any]:
    if not live:
        flags = mention_flags(candidate.explanation if condition == "shared" else candidate.task.requirement)
        return {
            "cases": [copy.deepcopy(candidate.task.visible_input)],
            "check_no_mutation": bool(flags["mentions_mutation"]),
            "notes": "offline deterministic smoke validator",
        }
    assert client is not None
    if condition == "shared":
        system = (
            "You are generating a small public smoke-validation plan for a Python function. "
            "Use the author's explanation as the current interpretation of the requirement, "
            "and choose one or two JSON-serializable inputs that check consistency with that interpretation. "
            "Return exactly JSON: {\"cases\":[...], \"check_no_mutation\": true/false, \"notes\":\"...\"}. "
            "Do not include expected outputs."
        )
        shared = f"\nAuthor explanation visible to you:\n{candidate.explanation}\n"
    else:
        system = (
            "You are generating a small public validation plan for a Python function from the original requirement. "
            "Choose up to three JSON-serializable inputs for solve(data), including important edge clauses when obvious. "
            "Return exactly JSON: {\"cases\":[...], \"check_no_mutation\": true/false, \"notes\":\"...\"}. "
            "Do not include expected outputs."
        )
        shared = "\nAuthor explanation is not available.\n"
    response = client.complete_json(
        [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": (
                    f"Requirement:\n{candidate.task.requirement}\n\nCode:\n{candidate.code}\n"
                    f"{shared}\nVisible example input:\n"
                    f"{json.dumps(candidate.task.visible_input, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=temperature,
    )
    cases = response.get("cases")
    if not isinstance(cases, list) or not cases:
        cases = [copy.deepcopy(candidate.task.visible_input)]
    return {
        "cases": cases[:3],
        "check_no_mutation": bool(response.get("check_no_mutation", False)),
        "notes": str(response.get("notes", "")),
    }


def validator_review(
    client: HostedChatClient | None,
    candidate: Candidate,
    condition: str,
    live: bool,
    temperature: float,
) -> Mapping[str, Any]:
    if not live:
        flags = mention_flags(candidate.explanation if condition == "shared" else candidate.task.requirement)
        return {
            "passes": True,
            "mentions_no_mutation": flags["mentions_mutation"],
            "mentions_edge": flags["mentions_edge"],
            "rationale": "offline deterministic smoke review",
        }
    assert client is not None
    if condition == "shared":
        system = (
            "You are reviewing a Python solve(data) implementation against the author's stated interpretation "
            "and the visible example. Return exactly JSON: {\"passes\": true/false, "
            "\"mentions_no_mutation\": true/false, \"mentions_edge\": true/false, \"rationale\":\"...\"}."
        )
        shared = f"\nAuthor explanation visible to you:\n{candidate.explanation}\n"
    else:
        system = (
            "You are reviewing a Python solve(data) implementation against the original requirement. "
            "Return exactly JSON: {\"passes\": true/false, \"mentions_no_mutation\": true/false, "
            "\"mentions_edge\": true/false, \"rationale\":\"...\"}."
        )
        shared = "\nAuthor explanation is not available.\n"
    response = client.complete_json(
        [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": f"Requirement:\n{candidate.task.requirement}\n\nCode:\n{candidate.code}\n{shared}",
            },
        ],
        temperature=temperature,
    )
    return {
        "passes": bool(response.get("passes", False)),
        "mentions_no_mutation": bool(response.get("mentions_no_mutation", False)),
        "mentions_edge": bool(response.get("mentions_edge", False)),
        "rationale": str(response.get("rationale", "")),
    }


def run_test_plan(candidate: Candidate, plan: Mapping[str, Any]) -> bool:
    raw_cases = plan.get("cases", [])
    cases = raw_cases if isinstance(raw_cases, list) else []
    if not cases:
        cases = [copy.deepcopy(candidate.task.visible_input)]
    valid_cases = []
    for case in cases:
        try:
            candidate.task.reference(copy.deepcopy(case))
        except Exception:
            continue
        valid_cases.append(case)
    if not valid_cases:
        valid_cases = [copy.deepcopy(candidate.task.visible_input)]
    return_ok = _return_passes(candidate, tuple(valid_cases[:3]))
    mutation_ok = True
    if bool(plan.get("check_no_mutation", False)):
        mutation_ok = _mutation_passes(candidate, tuple(valid_cases[:3]))
    return return_ok and mutation_ok


def make_observation(candidate: Candidate, condition: str, test_plan: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    cheap_test_pass = run_test_plan(candidate, test_plan)
    cheap_review_pass = bool(review.get("passes", False))
    tag = visible_tag(condition, candidate.explanation, test_plan, review)
    return {
        "task_id": candidate.task.task_id,
        "family": candidate.task.family,
        "condition": condition,
        "visible_tag": tag,
        "cheap_test_pass": cheap_test_pass,
        "cheap_review_pass": cheap_review_pass,
        "cheap_all_pass": cheap_test_pass and cheap_review_pass,
        "mutation_probe_pass": _mutation_passes(candidate, candidate.task.probe_inputs),
        "edge_probe_pass": _return_passes(candidate, candidate.task.probe_inputs),
        "final_correct": final_correct(candidate),
        "state": final_state(candidate),
        "explanation": candidate.explanation,
        "test_plan": dict(test_plan),
        "review": dict(review),
        "code": candidate.code,
        "live_generated": candidate.live_generated,
    }


def wilson_interval(successes: int, n: int, z: float = Z_ONE_SIDED_95) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - radius) / denom), min(1.0, (centre + radius) / denom)


def fit_point(records: list[dict[str, Any]], history: tuple[str, str]) -> dict[tuple[str, bool, bool], float]:
    matching = [
        row for row in records
        if row["condition"] == history[0]
        and row["visible_tag"] == history[1]
        and row["cheap_all_pass"]
    ]
    denominator = len(matching) + 0.5 * len(ATOM_SPACE)
    return {
        atom: (
            sum(
                row["state"] == atom[0]
                and row["mutation_probe_pass"] == atom[1]
                and row["edge_probe_pass"] == atom[2]
                for row in matching
            )
            + 0.5
        )
        / denominator
        for atom in ATOM_SPACE
    }


def fit_intervals(records: list[dict[str, Any]], history: tuple[str, str]) -> dict[tuple[str, bool, bool], tuple[float, float]]:
    matching = [
        row for row in records
        if row["condition"] == history[0]
        and row["visible_tag"] == history[1]
        and row["cheap_all_pass"]
    ]
    intervals = {}
    for atom in ATOM_SPACE:
        count = sum(
            row["state"] == atom[0]
            and row["mutation_probe_pass"] == atom[1]
            and row["edge_probe_pass"] == atom[2]
            for row in matching
        )
        intervals[atom] = wilson_interval(count, len(matching))
    if sum(low for low, _ in intervals.values()) > 1.0 or sum(high for _, high in intervals.values()) < 1.0:
        return {atom: (0.0, 1.0) for atom in ATOM_SPACE}
    return intervals


def maximize_linear(intervals: dict[tuple[str, bool, bool], tuple[float, float]], coefficient: Callable[[tuple[str, bool, bool]], float]) -> float:
    allocation = {atom: low for atom, (low, _) in intervals.items()}
    remaining = 1.0 - sum(allocation.values())
    for atom in sorted(intervals, key=coefficient, reverse=True):
        if remaining <= 1e-12:
            break
        low, high = intervals[atom]
        extra = min(high - low, remaining)
        allocation[atom] += extra
        remaining -= extra
    return sum(allocation[atom] * coefficient(atom) for atom in intervals)


def risk_point(distribution: dict[tuple[str, bool, bool], float]) -> float:
    return sum(prob for atom, prob in distribution.items() if atom[0] != "correct")


def risk_upper(intervals: dict[tuple[str, bool, bool], tuple[float, float]]) -> float:
    return maximize_linear(intervals, lambda atom: 1.0 if atom[0] != "correct" else 0.0)


def point_after_query(distribution: dict[tuple[str, bool, bool], float], query: str, passed: bool) -> float:
    index = 1 if query == "mutation_probe" else 2
    denominator = sum(prob for atom, prob in distribution.items() if atom[index] == passed)
    if denominator <= 0:
        return 1.0
    numerator = sum(prob for atom, prob in distribution.items() if atom[0] != "correct" and atom[index] == passed)
    return numerator / denominator


def credal_after_query(intervals: dict[tuple[str, bool, bool], tuple[float, float]], query: str, passed: bool) -> float:
    index = 1 if query == "mutation_probe" else 2
    relevant = {atom: bounds for atom, bounds in intervals.items() if atom[index] == passed}
    outside_lower = sum(low for atom, (low, _) in intervals.items() if atom[index] != passed)
    max_denominator = min(sum(high for _, high in relevant.values()), 1.0 - outside_lower)
    if max_denominator <= 1e-12:
        return 1.0
    wrong_upper = maximize_linear(relevant, lambda atom: 1.0 if atom[0] != "correct" else 0.0)
    return min(1.0, wrong_upper / max_denominator)


def point_query_loss(distribution: dict[tuple[str, bool, bool], float], query: str, delta: float = DELTA) -> float:
    index = 1 if query == "mutation_probe" else 2
    pass_probability = sum(prob for atom, prob in distribution.items() if atom[index])
    pass_risk = point_after_query(distribution, query, True)
    if pass_risk <= delta:
        return PROBE_COST + pass_probability * pass_risk + (1.0 - pass_probability) * DECLINE_LOSS
    return PROBE_COST + DECLINE_LOSS


def credal_query_loss(intervals: dict[tuple[str, bool, bool], tuple[float, float]], query: str, delta: float = DELTA) -> float:
    pass_risk = credal_after_query(intervals, query, True)
    if pass_risk > delta:
        return PROBE_COST + DECLINE_LOSS
    index = 1 if query == "mutation_probe" else 2
    return PROBE_COST + maximize_linear(
        intervals,
        lambda atom: (
            1.0 if atom[index] and atom[0] != "correct"
            else DECLINE_LOSS if not atom[index]
            else 0.0
        ),
    )


def select_point(distribution: dict[tuple[str, bool, bool], float], delta: float = DELTA) -> tuple[str, float]:
    risk = risk_point(distribution)
    if risk <= delta:
        return "accept", risk
    losses = {
        "reject": DECLINE_LOSS,
        "mutation_probe": point_query_loss(distribution, "mutation_probe", delta),
        "edge_probe": point_query_loss(distribution, "edge_probe", delta),
    }
    action, loss = min(losses.items(), key=lambda item: item[1])
    return (action if loss < DECLINE_LOSS else "reject"), risk


def select_credal(intervals: dict[tuple[str, bool, bool], tuple[float, float]]) -> tuple[str, float]:
    risk = risk_upper(intervals)
    if risk <= DELTA:
        return "accept", risk
    losses = {
        "reject": DECLINE_LOSS,
        "mutation_probe": credal_query_loss(intervals, "mutation_probe"),
        "edge_probe": credal_query_loss(intervals, "edge_probe"),
    }
    action, loss = min(losses.items(), key=lambda item: item[1])
    return (action if loss < DECLINE_LOSS else "reject"), risk


def apply_policy(
    row: dict[str, Any],
    policy: str,
    point: dict[tuple[str, bool, bool], float],
    intervals: dict[tuple[str, bool, bool], tuple[float, float]],
    bayes_delta: float = DELTA,
) -> dict[str, Any]:
    if not row["cheap_all_pass"]:
        return {"accepted": False, "cost": 0.0, "query": "none", "query_count": 0, "risk": 1.0}
    if policy in {"bayesian_point", "bayesian_cost_matched"}:
        action, risk = select_point(point, bayes_delta)
        if action == "accept":
            return {"accepted": True, "cost": 0.0, "query": "none", "query_count": 0, "risk": risk}
        if action == "reject":
            return {"accepted": False, "cost": 0.0, "query": "none", "query_count": 0, "risk": risk}
        passed = bool(row[action + "_pass"])
        post = point_after_query(point, action, passed)
        return {"accepted": passed and post <= bayes_delta, "cost": PROBE_COST, "query": action, "query_count": 1, "risk": risk}
    if policy == "dynamic_credal":
        action, risk = select_credal(intervals)
        if action == "accept":
            return {"accepted": True, "cost": 0.0, "query": "none", "query_count": 0, "risk": risk}
        if action == "reject":
            return {"accepted": False, "cost": 0.0, "query": "none", "query_count": 0, "risk": risk}
        passed = bool(row[action + "_pass"])
        post = credal_after_query(intervals, action, passed)
        return {"accepted": passed and post <= DELTA, "cost": PROBE_COST, "query": action, "query_count": 1, "risk": risk}
    if policy == "fixed_mutation_probe":
        return {
            "accepted": bool(row["mutation_probe_pass"]),
            "cost": PROBE_COST,
            "query": "mutation_probe",
            "query_count": 1,
            "risk": None,
        }
    if policy == "sequential_probes":
        if not bool(row["mutation_probe_pass"]):
            return {"accepted": False, "cost": PROBE_COST, "query": "mutation_probe", "query_count": 1, "risk": None}
        return {
            "accepted": bool(row["edge_probe_pass"]),
            "cost": 2.0 * PROBE_COST,
            "query": "mutation_probe+edge_probe",
            "query_count": 2,
            "risk": None,
        }
    raise ValueError(policy)


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    accepted = [row for row in rows if row["accepted"]]
    wrong = [row for row in accepted if not row["final_correct"]]
    correct = [row for row in accepted if row["final_correct"]]
    return {
        "tasks": len(rows),
        "accept_rate": mean(float(row["accepted"]) for row in rows) if rows else 0.0,
        "error_among_accepted": len(wrong) / len(accepted) if accepted else None,
        "wrong_accept_per_task": len(wrong) / len(rows) if rows else 0.0,
        "correct_completion_rate": len(correct) / len(rows) if rows else 0.0,
        "mean_cost": mean(float(row["cost"]) for row in rows) if rows else 0.0,
        "mean_probes": mean(float(row["query_count"]) for row in rows) if rows else 0.0,
    }


def replay(records: list[dict[str, Any]], calibration_ids: set[str], include_cost_matched: bool = True) -> dict[str, Any]:
    calibration = [row for row in records if row["task_id"] in calibration_ids]
    test_rows = [row for row in records if row["task_id"] not in calibration_ids]
    histories = sorted({(row["condition"], row["visible_tag"]) for row in records})
    fitted_point = {history: fit_point(calibration, history) for history in histories}
    fitted_intervals = {history: fit_intervals(calibration, history) for history in histories}
    policies = ("bayesian_point", "fixed_mutation_probe", "sequential_probes", "dynamic_credal")
    decision_rows: list[dict[str, Any]] = []
    threshold_by_condition = {"shared": DELTA, "independent": DELTA}
    if include_cost_matched:
        threshold_by_condition = select_cost_matched_thresholds(calibration, fitted_point, fitted_intervals)
        policies = policies + ("bayesian_cost_matched",)
    for row in test_rows:
        history = (row["condition"], row["visible_tag"])
        for policy in policies:
            decision = apply_policy(
                row,
                policy,
                fitted_point[history],
                fitted_intervals[history],
                threshold_by_condition[row["condition"]] if policy == "bayesian_cost_matched" else DELTA,
            )
            decision_rows.append({**row, **decision, "policy": policy})
    return {
        "summary": {
            condition: {
                policy: summarize([
                    row for row in decision_rows if row["condition"] == condition and row["policy"] == policy
                ])
                for policy in policies
            }
            for condition in ("shared", "independent")
        },
        "threshold_by_condition": threshold_by_condition,
        "decisions": decision_rows,
    }


def select_cost_matched_thresholds(
    calibration: list[dict[str, Any]],
    fitted_point: dict[tuple[str, str], dict[tuple[str, bool, bool], float]],
    fitted_intervals: dict[tuple[str, str], dict[tuple[str, bool, bool], tuple[float, float]]],
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for condition in ("shared", "independent"):
        rows = [row for row in calibration if row["condition"] == condition]
        credal_rows = []
        for row in rows:
            history = (row["condition"], row["visible_tag"])
            credal_rows.append({**row, **apply_policy(row, "dynamic_credal", fitted_point[history], fitted_intervals[history])})
        target = float(summarize(credal_rows)["mean_cost"])
        best = (float("inf"), DELTA)
        for threshold in POINT_THRESHOLD_GRID:
            point_rows = []
            for row in rows:
                history = (row["condition"], row["visible_tag"])
                point_rows.append({
                    **row,
                    **apply_policy(row, "bayesian_cost_matched", fitted_point[history], fitted_intervals[history], threshold),
                })
            cost = float(summarize(point_rows)["mean_cost"])
            score = (abs(cost - target), -threshold)
            if score < best:
                best = (score[0], threshold)
        thresholds[condition] = best[1]
    return thresholds


def cheap_failure_rates(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output = {}
    for condition in ("shared", "independent"):
        rows = [row for row in records if row["condition"] == condition]
        all_pass = [row for row in rows if row["cheap_all_pass"]]
        output[condition] = {
            "cheap_all_pass_rate": len(all_pass) / len(rows) if rows else 0.0,
            "wrong_given_all_pass": (
                sum(not row["final_correct"] for row in all_pass) / len(all_pass)
                if all_pass else 0.0
            ),
        }
    return output


def tex(value: str) -> str:
    return value.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmt(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def write_table(result: dict[str, Any], path: Path = TABLE_OUT) -> None:
    labels = {
        "bayesian_point": "Bayesian point",
        "fixed_mutation_probe": "Fixed mutation probe",
        "sequential_probes": "Sequential probes",
        "dynamic_credal": "Dynamic credal",
    }
    lines = [
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Condition & Policy & Accept & Error/accepted & Wrong accept & Correct completion & Cost\\",
        r"\midrule",
    ]
    for condition in ("shared", "independent"):
        for policy, label in labels.items():
            row = result["summary"][condition][policy]
            lines.append(
                "{} & {} & {} & {} & {} & {} & {}\\\\".format(
                    condition.title(),
                    label,
                    fmt(row["accept_rate"]),
                    fmt(row["error_among_accepted"]),
                    fmt(row["wrong_accept_per_task"]),
                    fmt(row["correct_completion_rate"]),
                    fmt(row["mean_cost"]),
                )
            )
        if condition == "shared":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_case_table(records: list[dict[str, Any]], decisions: list[dict[str, Any]], path: Path = CASE_OUT) -> None:
    by_key = {(row["task_id"], row["condition"], row["policy"]): row for row in decisions}
    examples = [
        row for row in records
        if row["condition"] == "shared" and row["cheap_all_pass"] and not row["final_correct"]
    ][:3]
    lines = [
        r"\begin{tabular}{@{}llllll@{}}",
        r"\toprule",
        r"Task & Family & State & Visible tag & Credal query & Outcome\\",
        r"\midrule",
    ]
    for row in examples:
        decision = by_key.get((row["task_id"], row["condition"], "dynamic_credal"), {})
        outcome = "accept" if decision.get("accepted") else "reject"
        lines.append(
            "{} & {} & {} & {} & {} & {}\\\\".format(
                tex(row["task_id"]),
                tex(row["family"]),
                tex(row["state"]),
                tex(row["visible_tag"]),
                tex(str(decision.get("query", "none"))),
                outcome,
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return ordered[index]


def family_split_replay(records: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({row["family"] for row in records})
    if len(families) < 2:
        raise ValueError("family split replay requires at least two task families")
    policies = ("bayesian_cost_matched", "sequential_probes", "dynamic_credal")
    rows_by_condition = {
        condition: {policy: [] for policy in policies}
        for condition in ("shared", "independent")
    }
    split_size = len(families) // 2
    for calibration_families in itertools.combinations(families, split_size):
        calibration_set = set(calibration_families)
        calibration_ids = {
            row["task_id"]
            for row in records
            if row["family"] in calibration_set
        }
        replayed = replay(records, calibration_ids, include_cost_matched=True)
        for condition in ("shared", "independent"):
            for policy in policies:
                rows_by_condition[condition][policy].append(replayed["summary"][condition][policy])
    summary: dict[str, dict[str, dict[str, float | None]]] = {}
    for condition in ("shared", "independent"):
        summary[condition] = {}
        for policy in policies:
            metrics: dict[str, float | None] = {}
            for key in (
                "accept_rate",
                "error_among_accepted",
                "wrong_accept_per_task",
                "correct_completion_rate",
                "mean_cost",
                "mean_probes",
            ):
                values = [row[key] for row in rows_by_condition[condition][policy] if row[key] is not None]
                metrics[f"{key}_mean"] = mean(values) if values else None
                metrics[f"{key}_lo"] = _percentile(values, 0.025) if values else None
                metrics[f"{key}_hi"] = _percentile(values, 0.975) if values else None
            summary[condition][policy] = metrics
    return {
        "split": f"all {split_size}-of-{len(families)} task-family calibration splits",
        "families": families,
        "num_splits": math.comb(len(families), split_size),
        "summary": summary,
    }


def write_family_split_table(result: dict[str, Any], path: Path = FAMILY_SPLIT_TABLE_OUT) -> None:
    labels = {
        "bayesian_cost_matched": "Bayes threshold",
        "sequential_probes": "Sequential probes",
        "dynamic_credal": "Dynamic credal",
    }
    def row(*cells: str) -> str:
        return " & ".join(cells) + r"\\"

    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        row("Policy", "Accept", "Error/accepted", "Wrong accept", "Correct completion", "Cost", "Probes"),
        r"\midrule",
    ]
    for policy in ("bayesian_cost_matched", "sequential_probes", "dynamic_credal"):
        metrics = result["summary"]["shared"][policy]
        lines.append(
            row(
                labels[policy],
                fmt(metrics["accept_rate_mean"]),
                fmt(metrics["error_among_accepted_mean"]),
                fmt(metrics["wrong_accept_per_task_mean"]),
                fmt(metrics["correct_completion_rate_mean"]),
                fmt(metrics["mean_cost_mean"]),
                fmt(metrics["mean_probes_mean"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    client = None
    if args.live:
        client = HostedChatClient(
            ChatAPIConfig(
                endpoint=args.endpoint,
                model=args.model,
                authorization_source=args.auth_source,
                timeout=args.timeout,
            ),
            cache_dir=args.cache_dir,
        )
    tasks = build_tasks(args.tasks)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    def observe_task(task: TaskSpec) -> list[dict[str, Any]]:
        local_client = client
        if args.live and args.workers > 1:
            local_client = HostedChatClient(
                ChatAPIConfig(
                    endpoint=args.endpoint,
                    model=args.model,
                    authorization_source=args.auth_source,
                    timeout=args.timeout,
                ),
                cache_dir=args.cache_dir,
            )
        candidate = generate_candidate(
            local_client,
            task,
            rng,
            args.live,
            args.candidate_temperature,
            args.quick_coder,
        )
        task_records: list[dict[str, Any]] = []
        for condition in ("shared", "independent"):
            test_plan = validator_test_plan(
                local_client,
                candidate,
                condition,
                args.live,
                args.validator_temperature,
            )
            review = validator_review(
                local_client,
                candidate,
                condition,
                args.live,
                args.validator_temperature,
            )
            task_records.append(make_observation(candidate, condition, test_plan, review))
        return task_records

    records: list[dict[str, Any]] = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for task_records in executor.map(observe_task, tasks):
                records.extend(task_records)
    else:
        for task in tasks:
            records.extend(observe_task(task))
    ids = sorted({row["task_id"] for row in records})
    rng.shuffle(ids)
    calibration_n = max(1, min(len(ids) - 1, round(len(ids) * args.calibration_fraction)))
    calibration_ids = set(ids[:calibration_n])
    replayed = replay(records, calibration_ids)
    result = {
        "protocol": {
            "version": "llm-shared-misunderstanding-v1",
            "live": bool(args.live),
            "model": args.model if args.live else "offline-smoke",
            "candidate_temperature": args.candidate_temperature if args.live else None,
            "validator_temperature": args.validator_temperature if args.live else None,
            "quick_coder": bool(args.quick_coder) if args.live else False,
            "seed": args.seed,
            "tasks": len(tasks),
            "calibration_tasks": calibration_n,
            "endpoint": args.endpoint if args.live else None,
            "credentials_stored": False,
        },
        "cheap_failure_rates": cheap_failure_rates(records),
        "replay": {key: value for key, value in replayed.items() if key != "decisions"},
        "records": records,
        "decisions": replayed["decisions"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Call the configured hosted model endpoint.")
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--endpoint", default=os.environ.get("OPENAI_COMPATIBLE_ENDPOINT", "https://api.openai.com/v1/chat/completions"))
    parser.add_argument("--auth-source", default=None)
    parser.add_argument("--cache-dir", default="experiments/cache/llm_shared_misunderstanding")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--tasks", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--candidate-temperature", type=float, default=0.7)
    parser.add_argument("--validator-temperature", type=float, default=0.1)
    parser.add_argument("--quick-coder", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--calibration-fraction", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--no-tex", action="store_true")
    args = parser.parse_args()
    if args.tasks < 2:
        raise ValueError("--tasks must be at least 2")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    RESULTS.mkdir(parents=True, exist_ok=True)
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    family_result = family_split_replay(result["records"])
    family_json_path = args.output.with_name(f"{args.output.stem}_family_split.json")
    family_json_path.write_text(json.dumps(family_result, indent=2, sort_keys=True), encoding="utf-8")
    if not args.no_tex:
        write_table(result["replay"])
        write_case_table(result["records"], result["decisions"])
        family_table_path = args.output.with_name(f"{args.output.stem}_family_split_table.tex")
        write_family_split_table(family_result, family_table_path)
        if args.model == "qwen/qwen-turbo" and args.tasks >= 240:
            FAMILY_SPLIT_OUT.write_text(json.dumps(family_result, indent=2, sort_keys=True), encoding="utf-8")
            write_family_split_table(family_result, FAMILY_SPLIT_TABLE_OUT)
    printable = {key: value for key, value in result.items() if key not in {"records", "decisions"}}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
