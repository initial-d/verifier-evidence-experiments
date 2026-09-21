#!/usr/bin/env python3
"""Cross-repository source-pattern fixture for shared verification.

This fixture adds non-packaging real repository checks to the mined-issue
evidence. It downloads pinned public snapshots, applies candidate patches with
git apply, and runs hidden behavioral checks after the policy choice is frozen.
The generated repairs are not official PR diffs.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "experiments" / "results"
JSON_OUT = RESULT_DIR / "cross_repo_source_pattern_fixture.json"
TEX_OUT = RESULT_DIR / "cross_repo_source_pattern_fixture_table.tex"
CACHE_DIR = ROOT / "experiments" / "cache" / "cross_repo_source_pattern_fixture"
ENV_PYTHON = ROOT / "external" / "agentdojo_env" / "bin" / "python"
PYTHON_BIN = str(ENV_PYTHON if ENV_PYTHON.exists() else Path(sys.executable))


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str


@dataclass(frozen=True)
class HiddenCheck:
    task_id: str
    description: str
    code: str


@dataclass(frozen=True)
class RepoFixture:
    fixture_id: str
    repo: str
    issue_url: str
    archive_sha: str
    source_file: str
    visible_contract: str
    candidates: tuple[Candidate, ...]
    hidden_checks: tuple[HiddenCheck, ...]
    policy_choice: dict[str, str]

    @property
    def archive_url(self) -> str:
        return f"https://github.com/{self.repo}/archive/{self.archive_sha}.zip"


FIXTURES = (
    RepoFixture(
        fixture_id="jsonpatch_137_mutable_patch_value",
        repo="stefankoegl/python-json-patch",
        issue_url="https://github.com/stefankoegl/python-json-patch/issues/137",
        archive_sha="d8e1a6e244728c04229d601bc9a384d9b034c603",
        source_file="jsonpatch.py",
        visible_contract=(
            "contract: applying the same JsonPatch object repeatedly must not "
            "reuse mutable operation values across target documents"
        ),
        candidates=(
            Candidate("no_patch", "none"),
            Candidate("generated_shallow_copy_value", "shallow_copy"),
            Candidate("generated_deepcopy_value", "deepcopy"),
        ),
        hidden_checks=(
            HiddenCheck(
                "jsonpatch_137_nested_dict_value_reuse",
                "nested dict value not reused",
                "\n".join([
                    "import jsonpatch",
                    "patch = jsonpatch.JsonPatch([{'op': 'add', 'path': '/items/0', 'value': {'nested': []}}])",
                    "first = {'items': []}",
                    "second = {'items': []}",
                    "out1 = patch.apply(first, in_place=False)",
                    "out1['items'][0]['nested'].append('x')",
                    "out2 = patch.apply(second, in_place=False)",
                    "assert out2 == {'items': [{'nested': []}]}, out2",
                ]),
            ),
            HiddenCheck(
                "jsonpatch_137_nested_list_value_reuse",
                "nested list value not reused",
                "\n".join([
                    "import jsonpatch",
                    "patch = jsonpatch.JsonPatch([{'op': 'add', 'path': '/items/0', 'value': [['seed']]}])",
                    "first = {'items': []}",
                    "second = {'items': []}",
                    "out1 = patch.apply(first, in_place=False)",
                    "out1['items'][0][0].append('x')",
                    "out2 = patch.apply(second, in_place=False)",
                    "assert out2 == {'items': [[['seed']]]}, out2",
                ]),
            ),
            HiddenCheck(
                "jsonpatch_137_scalar_add_still_works",
                "scalar add preserved",
                "\n".join([
                    "import jsonpatch",
                    "patch = jsonpatch.JsonPatch([{'op': 'add', 'path': '/items/0', 'value': 3}])",
                    "assert patch.apply({'items': []}, in_place=False) == {'items': [3]}",
                ]),
            ),
            HiddenCheck(
                "jsonpatch_137_nested_object_in_list_not_reused",
                "nested object inside list not reused",
                "\n".join([
                    "import jsonpatch",
                    "patch = jsonpatch.JsonPatch([{'op': 'add', 'path': '/items/0', 'value': [{'inner': []}]}])",
                    "first = {'items': []}",
                    "second = {'items': []}",
                    "out1 = patch.apply(first, in_place=False)",
                    "out1['items'][0][0]['inner'].append('x')",
                    "out2 = patch.apply(second, in_place=False)",
                    "assert out2 == {'items': [[{'inner': []}]]}, out2",
                ]),
            ),
            HiddenCheck(
                "jsonpatch_137_two_operation_values_independent",
                "two operation values remain independent",
                "\n".join([
                    "import jsonpatch",
                    "patch = jsonpatch.JsonPatch([",
                    "    {'op': 'add', 'path': '/left', 'value': {'items': []}},",
                    "    {'op': 'add', 'path': '/right', 'value': {'items': []}},",
                    "])",
                    "out1 = patch.apply({}, in_place=False)",
                    "out1['left']['items'].append('x')",
                    "out2 = patch.apply({}, in_place=False)",
                    "assert out2 == {'left': {'items': []}, 'right': {'items': []}}, out2",
                ]),
            ),
        ),
        policy_choice={
            "A_no_patch": "no_patch",
            "E_naive_shallow_pattern": "generated_shallow_copy_value",
            "F_source_pattern_generated": "generated_deepcopy_value",
        },
    ),
    RepoFixture(
        fixture_id="pyyaml_bool_resolver_yaml12_strings",
        repo="yaml/pyyaml",
        issue_url="https://github.com/yaml/pyyaml/issues/376",
        archive_sha="34a9bf82357f4952d8f194a5a31f1c39743652d0",
        source_file="lib/yaml/resolver.py",
        visible_contract=(
            "contract: YAML 1.2-like boolean resolution keeps legacy yes/no/on/off "
            "tokens as strings while preserving true/false booleans"
        ),
        candidates=(
            Candidate("no_patch", "none"),
            Candidate("generated_lowercase_onoff_resolver", "lowercase_onoff_resolver"),
            Candidate("generated_yaml12_bool_resolver", "yaml12_bool_resolver"),
        ),
        hidden_checks=(
            HiddenCheck(
                "pyyaml_376_on_off_are_strings",
                "on/off remain strings",
                "\n".join([
                    "import yaml",
                    "assert yaml.safe_load('flag: on')['flag'] == 'on'",
                    "assert yaml.safe_load('flag: Off')['flag'] == 'Off'",
                    "data = yaml.safe_load('on: 1\\ntrue: 2\\n')",
                    "assert 'on' in data and data['on'] == 1, data",
                    "assert True in data and data[True] == 2, data",
                ]),
            ),
            HiddenCheck(
                "pyyaml_376_yes_no_are_strings",
                "yes/no remain strings",
                "\n".join([
                    "import yaml",
                    "assert yaml.safe_load('flag: YES')['flag'] == 'YES'",
                    "assert yaml.safe_load('flag: no')['flag'] == 'no'",
                ]),
            ),
            HiddenCheck(
                "pyyaml_376_true_false_still_booleans",
                "true/false remain booleans",
                "\n".join([
                    "import yaml",
                    "assert yaml.safe_load('flag: true')['flag'] is True",
                    "assert yaml.safe_load('flag: FALSE')['flag'] is False",
                ]),
            ),
            HiddenCheck(
                "pyyaml_376_capitalized_yes_no_are_strings",
                "capitalized yes/no remain strings",
                "\n".join([
                    "import yaml",
                    "assert yaml.safe_load('flag: Yes')['flag'] == 'Yes'",
                    "assert yaml.safe_load('flag: NO')['flag'] == 'NO'",
                ]),
            ),
            HiddenCheck(
                "pyyaml_376_on_off_mapping_keys_are_strings",
                "on/off mapping keys remain strings",
                "\n".join([
                    "import yaml",
                    "data = yaml.safe_load('ON: 1\\noff: 2\\nfalse: 3\\n')",
                    "assert data['ON'] == 1, data",
                    "assert data['off'] == 2, data",
                    "assert data[False] == 3, data",
                ]),
            ),
        ),
        policy_choice={
            "A_no_patch": "no_patch",
            "E_naive_shallow_pattern": "generated_lowercase_onoff_resolver",
            "F_source_pattern_generated": "generated_yaml12_bool_resolver",
        },
    ),
    RepoFixture(
        fixture_id="click_2192_executable_path_flag",
        repo="pallets/click",
        issue_url="https://github.com/pallets/click/issues/2192",
        archive_sha="bf1a6d4956cbbbfd0a6f4dd6310c8110cf89f7fe",
        source_file="src/click/types.py",
        visible_contract=(
            "contract: click.Path accepts an executable flag and rejects "
            "non-executable existing paths when it is enabled"
        ),
        candidates=(
            Candidate("no_patch", "none"),
            Candidate("generated_accepts_executable_arg_only", "executable_arg_only"),
            Candidate("generated_executable_path_check", "executable_path_check"),
        ),
        hidden_checks=(
            HiddenCheck(
                "click_2192_rejects_non_executable",
                "reject non-executable file",
                "\n".join([
                    "import os, stat, tempfile",
                    "import click",
                    "td = tempfile.mkdtemp()",
                    "executable = os.path.join(td, 'run.sh')",
                    "plain = os.path.join(td, 'plain.txt')",
                    "with open(executable, 'w') as f: f.write('#!/bin/sh\\n')",
                    "with open(plain, 'w') as f: f.write('plain')",
                    "os.chmod(executable, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)",
                    "os.chmod(plain, stat.S_IRUSR | stat.S_IWUSR)",
                    "assert click.Path(exists=True, executable=True).convert(executable, None, None) == executable",
                    "try:",
                    "    click.Path(exists=True, executable=True).convert(plain, None, None)",
                    "except click.BadParameter:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('non-executable path accepted')",
                ]),
            ),
            HiddenCheck(
                "click_2192_missing_path_still_rejected",
                "missing path remains rejected when exists is true",
                "\n".join([
                    "import os, tempfile",
                    "import click",
                    "missing = os.path.join(tempfile.mkdtemp(), 'missing')",
                    "try:",
                    "    click.Path(exists=True, executable=True).convert(missing, None, None)",
                    "except click.BadParameter:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('missing path accepted')",
                ]),
            ),
            HiddenCheck(
                "click_2192_existing_file_default_still_works",
                "ordinary existing file paths still convert",
                "\n".join([
                    "import os, tempfile",
                    "import click",
                    "path = os.path.join(tempfile.mkdtemp(), 'plain.txt')",
                    "with open(path, 'w') as f: f.write('plain')",
                    "assert click.Path(exists=True).convert(path, None, None) == path",
                ]),
            ),
            HiddenCheck(
                "click_2192_resolve_path_still_rejects_non_executable",
                "resolve_path preserves executable rejection",
                "\n".join([
                    "import os, stat, tempfile",
                    "import click",
                    "td = tempfile.mkdtemp()",
                    "plain = os.path.join(td, 'plain.txt')",
                    "with open(plain, 'w') as f: f.write('plain')",
                    "os.chmod(plain, stat.S_IRUSR | stat.S_IWUSR)",
                    "try:",
                    "    click.Path(exists=True, executable=True, resolve_path=True).convert(plain, None, None)",
                    "except click.BadParameter:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('non-executable resolved path accepted')",
                ]),
            ),
            HiddenCheck(
                "click_2192_executable_file_with_readable_false",
                "executable check works with readable disabled",
                "\n".join([
                    "import os, stat, tempfile",
                    "import click",
                    "td = tempfile.mkdtemp()",
                    "run = os.path.join(td, 'run.sh')",
                    "plain = os.path.join(td, 'plain.txt')",
                    "with open(run, 'w') as f: f.write('#!/bin/sh\\n')",
                    "with open(plain, 'w') as f: f.write('plain')",
                    "os.chmod(run, stat.S_IRUSR | stat.S_IXUSR)",
                    "os.chmod(plain, stat.S_IRUSR | stat.S_IWUSR)",
                    "assert click.Path(exists=True, readable=False, executable=True).convert(run, None, None) == run",
                    "try:",
                    "    click.Path(exists=True, readable=False, executable=True).convert(plain, None, None)",
                    "except click.BadParameter:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('non-executable file accepted when readable=False')",
                ]),
            ),
        ),
        policy_choice={
            "A_no_patch": "no_patch",
            "E_naive_shallow_pattern": "generated_accepts_executable_arg_only",
            "F_source_pattern_generated": "generated_executable_path_check",
        },
    ),
    RepoFixture(
        fixture_id="jsonpointer_69_leading_zero_array_index",
        repo="stefankoegl/python-json-pointer",
        issue_url="https://github.com/stefankoegl/python-json-pointer/pull/69",
        archive_sha="97cd230d9c91b2a86a032c73c8127df34b1dcd16",
        source_file="jsonpointer.py",
        visible_contract=(
            "contract: JSON Pointer array indices are either exactly 0 or a "
            "nonzero digit followed by digits; object member names such as 01 "
            "must still work when the current document is a mapping"
        ),
        candidates=(
            Candidate("no_patch", "none"),
            Candidate("generated_start_anchor_only", "start_anchor_only"),
            Candidate("generated_array_index_full_anchor", "array_index_full_anchor"),
        ),
        hidden_checks=(
            HiddenCheck(
                "jsonpointer_69_rejects_leading_zero_indices",
                "leading-zero array indices rejected",
                "\n".join([
                    "from jsonpointer import JsonPointerException, resolve_pointer",
                    "doc = {'array': [10, 20, 30]}",
                    "for pointer in ['/array/01', '/array/00', '/array/001']:",
                    "    try:",
                    "        resolve_pointer(doc, pointer)",
                    "    except JsonPointerException:",
                    "        pass",
                    "    else:",
                    "        raise AssertionError(f'{pointer} accepted')",
                ]),
            ),
            HiddenCheck(
                "jsonpointer_69_valid_indices_preserved",
                "canonical array indices preserved",
                "\n".join([
                    "from jsonpointer import resolve_pointer",
                    "doc = {'array': [10, 20, 30]}",
                    "assert resolve_pointer(doc, '/array/0') == 10",
                    "assert resolve_pointer(doc, '/array/2') == 30",
                ]),
            ),
            HiddenCheck(
                "jsonpointer_69_mapping_member_preserved",
                "mapping member named 01 preserved",
                "\n".join([
                    "from jsonpointer import resolve_pointer",
                    "doc = {'array': {'01': 'member'}}",
                    "assert resolve_pointer(doc, '/array/01') == 'member'",
                ]),
            ),
            HiddenCheck(
                "jsonpointer_69_rejects_zero_padded_large_indices",
                "zero-padded array indices rejected",
                "\n".join([
                    "from jsonpointer import JsonPointerException, resolve_pointer",
                    "doc = {'array': [10, 20, 30]}",
                    "for pointer in ['/array/000', '/array/002']:",
                    "    try:",
                    "        resolve_pointer(doc, pointer)",
                    "    except JsonPointerException:",
                    "        pass",
                    "    else:",
                    "        raise AssertionError(f'{pointer} accepted')",
                ]),
            ),
            HiddenCheck(
                "jsonpointer_69_mapping_numeric_names_preserved",
                "mapping numeric-looking names preserved",
                "\n".join([
                    "from jsonpointer import resolve_pointer",
                    "doc = {'array': {'00': 'double-zero', '002': 'padded'}}",
                    "assert resolve_pointer(doc, '/array/00') == 'double-zero'",
                    "assert resolve_pointer(doc, '/array/002') == 'padded'",
                ]),
            ),
        ),
        policy_choice={
            "A_no_patch": "no_patch",
            "E_naive_shallow_pattern": "generated_start_anchor_only",
            "F_source_pattern_generated": "generated_array_index_full_anchor",
        },
    ),
    RepoFixture(
        fixture_id="jsonpointer_70_string_not_indexable",
        repo="stefankoegl/python-json-pointer",
        issue_url="https://github.com/stefankoegl/python-json-pointer/issues/70",
        archive_sha="5998f951dcc5ace60f67f35afe6778c445401a07",
        source_file="jsonpointer.py",
        visible_contract=(
            "contract: JSON Pointer tokens may index arrays and mappings, but "
            "strings are scalar values and must not be treated as indexable JSON arrays"
        ),
        candidates=(
            Candidate("no_patch", "none"),
            Candidate("generated_reject_zero_string_index", "reject_zero_string_index"),
            Candidate("generated_reject_all_string_indices", "reject_all_string_indices"),
        ),
        hidden_checks=(
            HiddenCheck(
                "jsonpointer_70_rejects_zero_string_index",
                "string token 0 rejected",
                "\n".join([
                    "from jsonpointer import JsonPointer, JsonPointerException, resolve_pointer",
                    "doc = {'foo': 'should-not-be-indexable'}",
                    "for resolver in (lambda: resolve_pointer(doc, '/foo/0'), lambda: JsonPointer('/foo/0').resolve(doc)):",
                    "    try:",
                    "        resolver()",
                    "    except JsonPointerException:",
                    "        pass",
                    "    else:",
                    "        raise AssertionError('string index 0 accepted')",
                ]),
            ),
            HiddenCheck(
                "jsonpointer_70_rejects_other_string_tokens",
                "other string tokens rejected",
                "\n".join([
                    "from jsonpointer import JsonPointerException, resolve_pointer",
                    "doc = {'foo': 'should-not-be-indexable'}",
                    "for pointer in ['/foo/1', '/foo/-']:",
                    "    try:",
                    "        resolve_pointer(doc, pointer)",
                    "    except JsonPointerException:",
                    "        pass",
                    "    else:",
                    "        raise AssertionError(f'{pointer} accepted')",
                ]),
            ),
            HiddenCheck(
                "jsonpointer_70_array_index_preserved",
                "array indexing preserved",
                "\n".join([
                    "from jsonpointer import resolve_pointer",
                    "doc = {'foo': ['a', 'b']}",
                    "assert resolve_pointer(doc, '/foo/0') == 'a'",
                    "assert resolve_pointer(doc, '/foo/1') == 'b'",
                ]),
            ),
            HiddenCheck(
                "jsonpointer_70_rejects_alpha_string_token",
                "alpha token on string rejected",
                "\n".join([
                    "from jsonpointer import JsonPointerException, resolve_pointer",
                    "doc = {'foo': 'should-not-be-indexable'}",
                    "try:",
                    "    resolve_pointer(doc, '/foo/a')",
                    "except JsonPointerException:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('string alpha token accepted')",
                ]),
            ),
            HiddenCheck(
                "jsonpointer_70_jsonpointer_object_rejects_nonzero_string_index",
                "JsonPointer object rejects nonzero string index",
                "\n".join([
                    "from jsonpointer import JsonPointer, JsonPointerException",
                    "doc = {'foo': 'should-not-be-indexable'}",
                    "try:",
                    "    JsonPointer('/foo/1').resolve(doc)",
                    "except JsonPointerException:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('JsonPointer accepted string index 1')",
                ]),
            ),
        ),
        policy_choice={
            "A_no_patch": "no_patch",
            "E_naive_shallow_pattern": "generated_reject_zero_string_index",
            "F_source_pattern_generated": "generated_reject_all_string_indices",
        },
    ),
)


POLICY_COST = {
    "A_no_patch": 0.05,
    "E_naive_shallow_pattern": 0.14,
    "F_source_pattern_generated": 0.19,
}


def _fetch(url: str) -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / hashlib.sha256(url.encode("utf-8")).hexdigest()
    if cache_path.exists():
        return cache_path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "credal-cross-repo-fixture/1.0"})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            cache_path.write_bytes(data)
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _extract_archive(directory: Path, archive: bytes, fixture: RepoFixture) -> Path:
    archive_path = directory / "base.zip"
    archive_path.write_bytes(archive)
    with zipfile.ZipFile(archive_path) as zipped:
        zipped.extractall(directory)
    prefix = fixture.repo.split("/")[-1]
    matches = sorted(directory.glob(f"{prefix}-*"))
    if not matches:
        raise RuntimeError(f"archive for {fixture.repo} did not contain {prefix}-*")
    return matches[0]


def _patch_from_text(root: Path, relpath: str, modified: str) -> str:
    target = root / relpath
    original = target.read_text(encoding="utf-8")
    tmp_original = root / "tmp_original.py"
    tmp_modified = root / "tmp_modified.py"
    tmp_original.write_text(original, encoding="utf-8")
    tmp_modified.write_text(modified, encoding="utf-8")
    diff = subprocess.run(
        ["git", "diff", "--no-index", "--", str(tmp_original), str(tmp_modified)],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    tmp_original.unlink()
    tmp_modified.unlink()
    diff = diff.replace(str(tmp_original), f"a/{relpath}")
    diff = diff.replace(str(tmp_modified), f"b/{relpath}")
    return diff


def _source_patch(root: Path, fixture: RepoFixture, candidate: Candidate) -> str:
    if candidate.kind == "none":
        return ""
    source = (root / fixture.source_file).read_text(encoding="utf-8")
    if fixture.fixture_id == "jsonpatch_137_mutable_patch_value":
        anchor = 'value = self.operation["value"]'
        if anchor not in source:
            raise RuntimeError(f"expected value-copy anchor missing in {fixture.source_file}")
        if candidate.kind == "shallow_copy":
            replacement = 'value = copy.copy(self.operation["value"])'
        elif candidate.kind == "deepcopy":
            replacement = 'value = copy.deepcopy(self.operation["value"])'
        else:
            raise ValueError(candidate.kind)
        return _patch_from_text(root, fixture.source_file, source.replace(anchor, replacement, 1))
    if fixture.fixture_id == "pyyaml_bool_resolver_yaml12_strings":
        full_bool_pattern = (
            "^(?:yes|Yes|YES|no|No|NO\n"
            "                    |true|True|TRUE|false|False|FALSE\n"
            "                    |on|On|ON|off|Off|OFF)$"
        )
        if full_bool_pattern not in source:
            raise RuntimeError(f"expected bool resolver anchor missing in {fixture.source_file}")
        if candidate.kind == "lowercase_onoff_resolver":
            modified = source.replace("|on|On|ON|off|Off|OFF", "", 1)
        elif candidate.kind == "yaml12_bool_resolver":
            modified = source.replace(
                full_bool_pattern,
                "^(?:true|True|TRUE|false|False|FALSE)$",
                1,
            )
        else:
            raise ValueError(candidate.kind)
        return _patch_from_text(root, fixture.source_file, modified)
    if fixture.fixture_id == "click_2192_executable_path_flag":
        def add_executable_argument(text: str) -> str:
            text = text.replace(
                "        readable: bool = True,\n        resolve_path: bool = False,",
                "        readable: bool = True,\n        executable: bool = False,\n        resolve_path: bool = False,",
                1,
            )
            text = text.replace(
                "        self.readable = readable\n        self.resolve_path = resolve_path",
                "        self.readable = readable\n        self.executable = executable\n        self.resolve_path = resolve_path",
                1,
            )
            return text

        modified = add_executable_argument(source)
        if candidate.kind == "executable_arg_only":
            return _patch_from_text(root, fixture.source_file, modified)
        if candidate.kind == "executable_path_check":
            readable_anchor = (
                "            if self.readable and not os.access(rv, os.R_OK):\n"
                "                self.fail(\n"
                "                    _(\"{name} {filename!r} is not readable.\").format(\n"
                "                        name=self.name.title(), filename=os.fsdecode(value)\n"
                "                    ),\n"
                "                    param,\n"
                "                    ctx,\n"
                "                )\n\n"
                "        return self.coerce_path_result(rv)\n"
            )
            if readable_anchor not in modified:
                raise RuntimeError(f"expected readable-check anchor missing in {fixture.source_file}")
            modified = modified.replace(
                readable_anchor,
                "            if self.readable and not os.access(rv, os.R_OK):\n"
                "                self.fail(\n"
                "                    _(\"{name} {filename!r} is not readable.\").format(\n"
                "                        name=self.name.title(), filename=os.fsdecode(value)\n"
                "                    ),\n"
                "                    param,\n"
                "                    ctx,\n"
                "                )\n"
                "            if self.executable and not os.access(rv, os.X_OK):\n"
                "                self.fail(\n"
                "                    _(\"{name} {filename!r} is not executable.\").format(\n"
                "                        name=self.name.title(), filename=os.fsdecode(value)\n"
                "                    ),\n"
                "                    param,\n"
                "                    ctx,\n"
                "                )\n\n"
                "        return self.coerce_path_result(rv)\n",
                1,
            )
            return _patch_from_text(root, fixture.source_file, modified)
        raise ValueError(candidate.kind)
    if fixture.fixture_id == "jsonpointer_69_leading_zero_array_index":
        anchor = "_RE_ARRAY_INDEX = re.compile('0|[1-9][0-9]*$')"
        if anchor not in source:
            raise RuntimeError(f"expected array-index regex anchor missing in {fixture.source_file}")
        if candidate.kind == "start_anchor_only":
            replacement = "_RE_ARRAY_INDEX = re.compile('^0|[1-9][0-9]*$')"
        elif candidate.kind == "array_index_full_anchor":
            replacement = "_RE_ARRAY_INDEX = re.compile('^(?:0|[1-9][0-9]*)$')"
        else:
            raise ValueError(candidate.kind)
        return _patch_from_text(root, fixture.source_file, source.replace(anchor, replacement, 1))
    if fixture.fixture_id == "jsonpointer_70_string_not_indexable":
        anchor = (
            "        if isinstance(doc, Mapping):\n"
            "            return part\n\n"
            "        elif isinstance(doc, Sequence):\n"
        )
        if anchor not in source:
            raise RuntimeError(f"expected mapping/sequence anchor missing in {fixture.source_file}")
        if candidate.kind == "reject_zero_string_index":
            inserted = (
                "        if isinstance(doc, Mapping):\n"
                "            return part\n\n"
                "        elif isinstance(doc, str) and str(part) == '0':\n"
                "            raise JsonPointerException(\"Cannot apply token '%s' to non-container type %s\" % (part, type(doc)))\n\n"
                "        elif isinstance(doc, Sequence):\n"
            )
        elif candidate.kind == "reject_all_string_indices":
            inserted = (
                "        if isinstance(doc, Mapping):\n"
                "            return part\n\n"
                "        elif isinstance(doc, str):\n"
                "            raise JsonPointerException(\"Cannot apply token '%s' to non-container type %s\" % (part, type(doc)))\n\n"
                "        elif isinstance(doc, Sequence):\n"
            )
        else:
            raise ValueError(candidate.kind)
        return _patch_from_text(root, fixture.source_file, source.replace(anchor, inserted, 1))
    raise ValueError(fixture.fixture_id)


def _apply_candidate_and_test(
    fixture: RepoFixture,
    check: HiddenCheck,
    candidate: Candidate,
    archive: bytes,
) -> tuple[bool, str, str]:
    with tempfile.TemporaryDirectory(prefix="credal-cross-repo-") as tmp:
        root = _extract_archive(Path(tmp), archive, fixture)
        patch_text = _source_patch(root, fixture, candidate)
        patch_hash = _sha256(patch_text)[:16]
        if patch_text:
            patch_file = root / "candidate.patch"
            patch_file.write_text(patch_text, encoding="utf-8")
            applied = subprocess.run(
                ["git", "apply", str(patch_file)],
                cwd=root,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
            )
            if applied.returncode != 0:
                return False, patch_hash, f"git_apply_failed:{applied.stderr[:120]}"
        test_file = root / "hidden_cross_repo_test.py"
        test_file.write_text(check.code + "\n", encoding="utf-8")
        env = os.environ.copy()
        if (root / "lib" / "yaml").exists():
            import_root = root / "lib"
        elif (root / "src" / "click").exists():
            import_root = root / "src"
        else:
            import_root = root
        env["PYTHONPATH"] = str(import_root)
        completed = subprocess.run(
            [PYTHON_BIN, str(test_file)],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        return completed.returncode == 0, patch_hash, (completed.stderr or completed.stdout)[-240:]


def _choose(fixture: RepoFixture, policy: str) -> Candidate:
    chosen = fixture.policy_choice[policy]
    return next(candidate for candidate in fixture.candidates if candidate.name == chosen)


def _paired(rows: list[dict[str, object]], metric: str, left: str, right: str) -> list[float]:
    by_task: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), {})[str(row["policy"])] = row
    return [
        float(policies[left][metric]) - float(policies[right][metric])
        for policies in by_task.values()
        if left in policies and right in policies
    ]


def run() -> dict[str, object]:
    policies = tuple(POLICY_COST)
    rows: list[dict[str, object]] = []
    fixtures = []
    for fixture in FIXTURES:
        archive = _fetch(fixture.archive_url)
        fixture_record = {
            "fixture_id": fixture.fixture_id,
            "repo": fixture.repo,
            "issue_url": fixture.issue_url,
            "archive_url": fixture.archive_url,
            "archive_sha": fixture.archive_sha,
            "archive_sha256": _sha256(archive),
            "source_file": fixture.source_file,
            "visible_contract": fixture.visible_contract,
            "hidden_checks": len(fixture.hidden_checks),
            "candidate_names": [candidate.name for candidate in fixture.candidates],
            "generated_candidates": [
                candidate.name for candidate in fixture.candidates if candidate.kind != "none"
            ],
            "generated_official_only": 0.0,
        }
        fixtures.append(fixture_record)
        for check in fixture.hidden_checks:
            outcomes = {}
            patch_hashes = {}
            errors = {}
            for candidate in fixture.candidates:
                passed, patch_hash, error = _apply_candidate_and_test(fixture, check, candidate, archive)
                outcomes[candidate.name] = passed
                patch_hashes[candidate.name] = patch_hash
                errors[candidate.name] = error
            best = float(any(outcomes.values()))
            for policy in policies:
                chosen = _choose(fixture, policy)
                success = float(outcomes[chosen.name])
                rows.append({
                    "task_id": check.task_id,
                    "fixture_id": fixture.fixture_id,
                    "repo": fixture.repo,
                    "issue_url": fixture.issue_url,
                    "policy": policy,
                    "chosen": chosen.name,
                    "patch_hash": patch_hashes[chosen.name],
                    "success": success,
                    "selection_loss": best - success,
                    "cost": POLICY_COST[policy],
                    "hidden_check": check.description,
                    "error": errors[chosen.name],
                })
    summary = {}
    for policy in policies:
        subset = [row for row in rows if row["policy"] == policy]
        summary[policy] = {
            "tasks": len(subset),
            "success_rate": mean(row["success"] for row in subset),
            "selection_loss": mean(row["selection_loss"] for row in subset),
            "mean_cost": mean(row["cost"] for row in subset),
        }
    paired = {}
    for left, right in [
        ("F_source_pattern_generated", "A_no_patch"),
        ("F_source_pattern_generated", "E_naive_shallow_pattern"),
    ]:
        paired[f"{left}_minus_{right}"] = {
            "success_diff": mean(_paired(rows, "success", left, right)),
            "utility_gap_reduction": mean(_paired(rows, "selection_loss", right, left)),
            "cost_diff": summary[left]["mean_cost"] - summary[right]["mean_cost"],
        }
    result = {
        "suite": "cross_repo_source_pattern_fixture",
        "scope": "non-packaging real public repository snapshots; hidden checks derived from public issue behavior",
        "python_bin": PYTHON_BIN,
        "fixture_count": len(FIXTURES),
        "task_count": len({row["task_id"] for row in rows}),
        "policies": list(policies),
        "fixtures": fixtures,
        "records": rows,
        "summary": summary,
        "paired": paired,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_tex(result)
    return result


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _write_tex(result: dict[str, object]) -> None:
    labels = {
        "A_no_patch": "A: no patch",
        "E_naive_shallow_pattern": "E: incomplete pattern",
        "F_source_pattern_generated": "F: source-pattern repair",
    }
    lines: list[str] = []
    lines.extend([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Cross-repository source-pattern fixture. The runner downloads pinned public snapshots, applies generated candidate patches with \texttt{git apply}, and runs hidden behavior checks after policy choice. Policy F uses non-official source-pattern repairs; policy E is a deliberately plausible but incomplete visible pattern. This is a four-repository development extension, not a benchmark.}",
        r"\label{tab:cross-repo-source-pattern}",
        r"\small",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Policy & Success & Cost & Utility gap \\",
        r"\midrule",
    ])
    summary = result["summary"]
    for key in ["A_no_patch", "E_naive_shallow_pattern", "F_source_pattern_generated"]:
        row = summary[key]
        lines.append(
            f"{labels[key]} & {_fmt(row['success_rate'])} & {_fmt(row['mean_cost'])} & "
            f"{_fmt(row['selection_loss'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
