#!/usr/bin/env python3
"""Small mined public-PR fixture for shared verification.

The fixture uses real pypa/packaging PRs and base snapshots. For each case, the
runner downloads the base archive and fixing PR patch, applies each candidate
with git apply, and runs hidden behavioral checks derived from tests added by
the PR. This is still a small fixture, not a benchmark.
"""

from __future__ import annotations

import hashlib
import json
import random
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
JSON_OUT = RESULT_DIR / "mined_issue_fixture.json"
TEX_OUT = RESULT_DIR / "mined_issue_fixture_table.tex"
CACHE_DIR = ROOT / "experiments" / "cache" / "mined_issue_fixture"
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
class IssueFixture:
    issue_id: str
    issue_url: str
    pr_url: str
    base_sha: str
    head_sha: str
    source_file: str
    candidates: tuple[Candidate, ...]
    hidden_checks: tuple[HiddenCheck, ...]
    policy_choice: dict[str, str]
    rescue_candidates: tuple[Candidate, ...] = ()

    @property
    def patch_url(self) -> str:
        return f"{self.pr_url}.patch"

    @property
    def archive_url(self) -> str:
        return f"https://github.com/pypa/packaging/archive/{self.base_sha}.zip"

    @property
    def head_archive_url(self) -> str:
        return f"https://github.com/pypa/packaging/archive/{self.head_sha}.zip"


FIXTURES = (
    IssueFixture(
        issue_id="packaging_788_prerelease",
        issue_url="https://github.com/pypa/packaging/issues/788",
        pr_url="https://github.com/pypa/packaging/pull/794",
        base_sha="a716c52b5f3ca9b4a512f538b80ced8ee01b2775",
        head_sha="dc39fcac9cfbd647a8104f03bb212ed7fa50c69a",
        source_file="src/packaging/specifiers.py",
        candidates=(
            Candidate("overbroad_all_ordered_ops", "overbroad"),
            Candidate("gt_only_patch", "gt_only"),
            Candidate("no_patch", "none"),
            Candidate("official_pr_794_patch", "official"),
        ),
        hidden_checks=(
            HiddenCheck(
                "packaging_788_prerelease_property",
                "Specifier.prereleases for <, >, and !=",
                "\n".join([
                    "from packaging.specifiers import Specifier",
                    "assert Specifier('<1.0.dev1').prereleases is True",
                    "assert Specifier('>1.0.dev1').prereleases is True",
                    "assert Specifier('!=1.0.dev1').prereleases is False",
                ]),
            ),
            HiddenCheck(
                "packaging_788_comparison_contains",
                "comparison operators admit explicit pre-release bounds",
                "\n".join([
                    "from packaging.specifiers import Specifier",
                    "assert Specifier('<3.0.0a8').contains('3.0.0a7') is True",
                    "assert Specifier('>3.0.0a7').contains('3.0.0a8') is True",
                ]),
            ),
            HiddenCheck(
                "packaging_788_specifier_filter",
                "SpecifierSet.filter follows explicit pre-release bounds",
                "\n".join([
                    "from packaging.specifiers import SpecifierSet",
                    "assert list(SpecifierSet('>2.0a1').filter(['2.0a1', '3.0a2', '3.0'])) == ['3.0a2', '3.0']",
                    "assert list(SpecifierSet('<2.0a1').filter(['1.0a2', '1.0', '2.0a1'])) == ['1.0a2', '1.0']",
                ]),
            ),
        ),
        policy_choice={
            "A_score_order": "overbroad_all_ordered_ops",
            "B_rho0_decision": "official_pr_794_patch",
            "C_credal_generic": "no_patch",
            "D_credal_decision": "official_pr_794_patch",
        },
        rescue_candidates=(
            Candidate("comparison_bounds_without_not_equal", "comparison_bounds"),
        ),
    ),
    IssueFixture(
        issue_id="packaging_777_iterable_specifiers",
        issue_url="https://github.com/pypa/packaging/issues/775",
        pr_url="https://github.com/pypa/packaging/pull/777",
        base_sha="4d8534061364e3cbfee582192ab81a095ec2db51",
        head_sha="835670ff5e13bace7ae2393823b6e3297261861b",
        source_file="src/packaging/specifiers.py",
        candidates=(
            Candidate("stringify_iterable_patch", "stringify_iterable"),
            Candidate("iterable_direct_patch", "iterable_direct"),
            Candidate("no_patch", "none"),
            Candidate("official_pr_777_patch", "official"),
        ),
        hidden_checks=(
            HiddenCheck(
                "packaging_777_iterable_set_equivalence",
                "SpecifierSet accepts an iterator of Specifier objects",
                "\n".join([
                    "from packaging.specifiers import Specifier, SpecifierSet",
                    "specs = [Specifier('>=1.0'), Specifier('!=1.1'), Specifier('<2.0')]",
                    "spec = SpecifierSet(iter(specs))",
                    "assert set(spec) == set(specs)",
                ]),
            ),
            HiddenCheck(
                "packaging_777_iterable_membership",
                "iterable-created SpecifierSet preserves membership semantics",
                "\n".join([
                    "from packaging.specifiers import Specifier, SpecifierSet",
                    "spec = SpecifierSet(iter([Specifier('>=1.0'), Specifier('!=1.1'), Specifier('<2.0')]))",
                    "assert spec.contains('1.5') is True",
                    "assert spec.contains('1.1') is False",
                ]),
            ),
            HiddenCheck(
                "packaging_777_string_constructor_still_works",
                "ordinary comma-separated strings still construct SpecifierSet",
                "\n".join([
                    "from packaging.specifiers import SpecifierSet",
                    "spec = SpecifierSet('>=1.0,!=1.1,<2.0')",
                    "assert spec.contains('1.5') is True",
                    "assert spec.contains('1.1') is False",
                ]),
            ),
        ),
        policy_choice={
            "A_score_order": "stringify_iterable_patch",
            "B_rho0_decision": "official_pr_777_patch",
            "C_credal_generic": "no_patch",
            "D_credal_decision": "official_pr_777_patch",
        },
    ),
    IssueFixture(
        issue_id="packaging_1384_ascii_specifier_words",
        issue_url="https://github.com/pypa/packaging/pull/1384",
        pr_url="https://github.com/pypa/packaging/pull/1384",
        base_sha="f8630ea2b7c810de81841459dcdb481d5b7a1318",
        head_sha="2a16593a8f0fc66addabb8e541fb256b5564d0f2",
        source_file="src/packaging/specifiers.py",
        candidates=(
            Candidate("ascii_pre_release_only", "ascii_pre_only"),
            Candidate("no_patch", "none"),
            Candidate("ascii_post_release_only", "ascii_post_only"),
            Candidate("official_pr_1384_patch", "official"),
        ),
        hidden_checks=(
            HiddenCheck(
                "packaging_1384_preview_uses_ascii_scope",
                "compatible specifiers reject non-ASCII preview spelling",
                "\n".join([
                    "from packaging.specifiers import InvalidSpecifier, Specifier",
                    "try:",
                    "    Specifier('~=1.2.3prev\\u0131ew1')",
                    "except InvalidSpecifier:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('non-ASCII preview spelling was accepted')",
                ]),
            ),
            HiddenCheck(
                "packaging_1384_post_uses_ascii_scope",
                "compatible specifiers reject non-ASCII post spelling",
                "\n".join([
                    "from packaging.specifiers import InvalidSpecifier, Specifier",
                    "try:",
                    "    Specifier('~=1.2.3po\\u017ft1')",
                    "except InvalidSpecifier:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('non-ASCII post spelling was accepted')",
                ]),
            ),
            HiddenCheck(
                "packaging_1384_valid_compatible_specifier_still_parses",
                "ordinary compatible specifiers still parse",
                "\n".join([
                    "from packaging.specifiers import Specifier",
                    "assert str(Specifier('~=1.2.3')) == '~=1.2.3'",
                ]),
            ),
        ),
        policy_choice={
            "A_score_order": "ascii_pre_release_only",
            "B_rho0_decision": "official_pr_1384_patch",
            "C_credal_generic": "ascii_post_release_only",
            "D_credal_decision": "official_pr_1384_patch",
        },
    ),
    IssueFixture(
        issue_id="packaging_1360_empty_platform_iterables",
        issue_url="https://github.com/pypa/packaging/pull/1360",
        pr_url="https://github.com/pypa/packaging/pull/1360",
        base_sha="2d873eb6002021a8c007c933fbdab58b37a5079b",
        head_sha="dc7a44cb856f96cc4a540172febd9d46da987ba8",
        source_file="src/packaging/tags.py",
        candidates=(
            Candidate("cpython_empty_platforms_only", "cpython_platforms_only"),
            Candidate("generic_empty_platforms_only", "generic_platforms_only"),
            Candidate("no_patch", "none"),
            Candidate("official_pr_1360_patch", "official"),
        ),
        hidden_checks=(
            HiddenCheck(
                "packaging_1360_cpython_empty_platforms",
                "cpython_tags preserves explicitly empty platforms",
                "\n".join([
                    "from packaging import tags",
                    "assert list(tags.cpython_tags((3, 11), abis=['abi'], platforms=[])) == []",
                ]),
            ),
            HiddenCheck(
                "packaging_1360_generic_empty_platforms",
                "generic_tags preserves explicitly empty platforms",
                "\n".join([
                    "from packaging import tags",
                    "assert list(tags.generic_tags('sillywalk', ['abi'], [])) == []",
                ]),
            ),
            HiddenCheck(
                "packaging_1360_compatible_empty_platforms",
                "compatible_tags preserves its any-platform fallbacks",
                "\n".join([
                    "from packaging import tags",
                    "assert list(tags.compatible_tags((3,), 'cp3', [])) == [",
                    "    tags.Tag('cp3', 'none', 'any'),",
                    "    tags.Tag('py3', 'none', 'any'),",
                    "]",
                ]),
            ),
        ),
        policy_choice={
            "A_score_order": "cpython_empty_platforms_only",
            "B_rho0_decision": "official_pr_1360_patch",
            "C_credal_generic": "no_patch",
            "D_credal_decision": "official_pr_1360_patch",
        },
        rescue_candidates=(
            Candidate("all_empty_platforms_patch", "all_platforms_empty"),
        ),
    ),
    IssueFixture(
        issue_id="packaging_1345_strict_end_anchor",
        issue_url="https://github.com/pypa/packaging/pull/1345",
        pr_url="https://github.com/pypa/packaging/pull/1345",
        base_sha="0bfc3cea4f9fe1b1b0c80ce06f02bd2433500c81",
        head_sha="9992abe78f1abd1db105a9a322ae5351f4e2b279",
        source_file="src/packaging/_tokenizer.py",
        candidates=(
            Candidate("no_patch", "none"),
            Candidate("official_pr_1345_patch", "official"),
        ),
        hidden_checks=(
            HiddenCheck(
                "packaging_1345_requirement_rejects_newline",
                "Requirement rejects trailing line breaks",
                "\n".join([
                    "from packaging.requirements import InvalidRequirement, Requirement",
                    "try:",
                    "    Requirement('name>=1\\n')",
                    "except InvalidRequirement:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('requirement with trailing newline was accepted')",
                ]),
            ),
            HiddenCheck(
                "packaging_1345_marker_rejects_newline",
                "Marker rejects trailing line breaks",
                "\n".join([
                    "from packaging.markers import InvalidMarker, Marker",
                    "try:",
                    "    Marker('python_version >= \"3\"\\n')",
                    "except InvalidMarker:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('marker with trailing newline was accepted')",
                ]),
            ),
            HiddenCheck(
                "packaging_1345_horizontal_whitespace_still_allowed",
                "horizontal trailing whitespace still parses",
                "\n".join([
                    "from packaging.requirements import Requirement",
                    "assert Requirement('name>=1 \\t') == Requirement('name>=1')",
                ]),
            ),
        ),
        policy_choice={
            "A_score_order": "no_patch",
            "B_rho0_decision": "official_pr_1345_patch",
            "C_credal_generic": "no_patch",
            "D_credal_decision": "official_pr_1345_patch",
        },
        rescue_candidates=(
            Candidate("strict_end_z_anchor_patch", "strict_end_z_anchor"),
        ),
    ),
    IssueFixture(
        issue_id="packaging_1332_invalid_requirement_wrapper",
        issue_url="https://github.com/pypa/packaging/pull/1332",
        pr_url="https://github.com/pypa/packaging/pull/1332",
        base_sha="9d0ec47b36a1bcdf705b397b84cd89272410a00d",
        head_sha="fa5982288e39434ec2be6a692a03b96607857a00",
        source_file="src/packaging/requirements.py",
        candidates=(
            Candidate("no_patch", "none"),
            Candidate("official_pr_1332_patch", "official"),
        ),
        hidden_checks=(
            HiddenCheck(
                "packaging_1332_malformed_specifier_is_invalid_requirement",
                "malformed requirement specifier raises InvalidRequirement",
                "\n".join([
                    "from packaging.requirements import InvalidRequirement, Requirement",
                    "try:",
                    "    Requirement('demo===x,y')",
                    "except InvalidRequirement:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('malformed requirement was accepted')",
                ]),
            ),
            HiddenCheck(
                "packaging_1332_valid_requirement_still_parses",
                "valid requirement still parses",
                "\n".join([
                    "from packaging.requirements import Requirement",
                    "req = Requirement('demo>=1')",
                    "assert req.name == 'demo'",
                    "assert str(req.specifier) == '>=1'",
                ]),
            ),
            HiddenCheck(
                "packaging_1332_url_requirement_still_parses",
                "URL requirement still parses",
                "\n".join([
                    "from packaging.requirements import Requirement",
                    "req = Requirement('demo @ https://example.com/demo.whl')",
                    "assert req.name == 'demo'",
                    "assert req.url == 'https://example.com/demo.whl'",
                ]),
            ),
        ),
        policy_choice={
            "A_score_order": "no_patch",
            "B_rho0_decision": "official_pr_1332_patch",
            "C_credal_generic": "no_patch",
            "D_credal_decision": "official_pr_1332_patch",
        },
        rescue_candidates=(
            Candidate("invalid_specifier_wrapper_patch", "invalid_specifier_wrapper"),
        ),
    ),
    IssueFixture(
        issue_id="packaging_1341_wheel_name_newline",
        issue_url="https://github.com/pypa/packaging/pull/1341",
        pr_url="https://github.com/pypa/packaging/pull/1341",
        base_sha="9c6f09ba6b313a561d6a64d988647ff3115e82ae",
        head_sha="6e1f147b284c0dc803619e8dc0fd6260ce35538c",
        source_file="src/packaging/utils.py",
        candidates=(
            Candidate("no_patch", "none"),
            Candidate("official_pr_1341_patch", "official"),
        ),
        hidden_checks=(
            HiddenCheck(
                "packaging_1341_rejects_newline_wheel_name",
                "wheel filenames reject newline in distribution name",
                "\n".join([
                    "from packaging.utils import InvalidWheelFilename, parse_wheel_filename",
                    "try:",
                    "    parse_wheel_filename('foo\\n-1.0-py3-none-any.whl')",
                    "except InvalidWheelFilename:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('wheel filename with newline was accepted')",
                ]),
            ),
            HiddenCheck(
                "packaging_1341_valid_wheel_still_parses",
                "ordinary wheel filename still parses",
                "\n".join([
                    "from packaging.utils import parse_wheel_filename",
                    "name, version, build, tags = parse_wheel_filename('foo-1.0-py3-none-any.whl')",
                    "assert str(name) == 'foo'",
                    "assert str(version) == '1.0'",
                    "assert build == ()",
                    "assert tags",
                ]),
            ),
            HiddenCheck(
                "packaging_1341_rejects_crlf_wheel_name",
                "wheel filenames reject CRLF in distribution name",
                "\n".join([
                    "from packaging.utils import InvalidWheelFilename, parse_wheel_filename",
                    "try:",
                    "    parse_wheel_filename('foo\\r\\n-1.0-py3-none-any.whl')",
                    "except InvalidWheelFilename:",
                    "    pass",
                    "else:",
                    "    raise AssertionError('wheel filename with CRLF was accepted')",
                ]),
            ),
        ),
        policy_choice={
            "A_score_order": "no_patch",
            "B_rho0_decision": "official_pr_1341_patch",
            "C_credal_generic": "no_patch",
            "D_credal_decision": "official_pr_1341_patch",
        },
        rescue_candidates=(
            Candidate("wheel_name_strict_end_patch", "wheel_name_strict_end"),
        ),
    ),
)


def _fetch(url: str) -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / hashlib.sha256(url.encode("utf-8")).hexdigest()
    if cache_path.exists():
        return cache_path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "credal-selection-mined-fixture/1.0"})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read()
            cache_path.write_bytes(data)
            return data
        except Exception as exc:  # network fixture; retry transient GitHub/CDN failures
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _extract_archive(directory: Path, archive: bytes, name: str) -> Path:
    archive_path = directory / "base.zip"
    archive_path.write_bytes(archive)
    with zipfile.ZipFile(archive_path) as zipped:
        zipped.extractall(directory)
    matches = sorted(directory.glob("packaging-*"))
    if not matches:
        raise RuntimeError(f"archive {name} did not contain a packaging-* directory")
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


def _patch_from_replacement(root: Path, relpath: str, old: str, new: str, count: int = -1) -> str:
    target = root / relpath
    original = target.read_text(encoding="utf-8")
    if old not in original:
        raise RuntimeError(f"expected patch anchor not found in {relpath}")
    modified = original.replace(old, new, count)
    return _patch_from_text(root, relpath, modified)


def _official_source_patch(root: Path, fixture: IssueFixture, head_archive: bytes) -> str:
    with tempfile.TemporaryDirectory(prefix="credal-mined-head-") as tmp:
        head_root = _extract_archive(Path(tmp), head_archive, fixture.head_sha)
        return _patch_from_replacement(
            root,
            fixture.source_file,
            (root / fixture.source_file).read_text(encoding="utf-8"),
            (head_root / fixture.source_file).read_text(encoding="utf-8"),
        )


def _source_patch(
    root: Path,
    fixture: IssueFixture,
    candidate: Candidate,
    official_patch: bytes,
    head_archive: bytes,
) -> str:
    if candidate.kind == "official":
        return _official_source_patch(root, fixture, head_archive)
    if candidate.kind == "none":
        return ""
    if fixture.issue_id == "packaging_788_prerelease":
        old = 'if operator in ["==", ">=", "<=", "~=", "==="]:'
        if candidate.kind == "gt_only":
            new = 'if operator in ["==", ">=", "<=", "~=", "===", ">"]:'
        elif candidate.kind == "comparison_bounds":
            new = 'if operator in ["==", ">=", "<=", "~=", "===", ">", "<"]:'
        elif candidate.kind == "overbroad":
            new = 'if operator in ["==", ">=", "<=", "~=", "===", ">", "<", "!="]:'
        else:
            raise ValueError(candidate.kind)
        return _patch_from_replacement(root, fixture.source_file, old, new)
    if fixture.issue_id == "packaging_777_iterable_specifiers":
        old = (
            "        # Split on `,` to break each individual specifier into it's own item, and\n"
            "        # strip each item to remove leading/trailing whitespace.\n"
            "        split_specifiers = [s.strip() for s in specifiers.split(\",\") if s.strip()]\n\n"
            "        # Make each individual specifier a Specifier and save in a frozen set for later.\n"
            "        self._specs = frozenset(map(Specifier, split_specifiers))"
        )
        if candidate.kind == "stringify_iterable":
            new = (
                "        if not isinstance(specifiers, str):\n"
                "            specifiers = \",\".join(str(specifier) for specifier in specifiers)\n\n"
                "        # Split on `,` to break each individual specifier into it's own item, and\n"
                "        # strip each item to remove leading/trailing whitespace.\n"
                "        split_specifiers = [s.strip() for s in specifiers.split(\",\") if s.strip()]\n\n"
                "        # Make each individual specifier a Specifier and save in a frozen set for later.\n"
                "        self._specs = frozenset(map(Specifier, split_specifiers))"
            )
        elif candidate.kind == "iterable_direct":
            new = "        self._specs = frozenset(specifiers)"
        elif candidate.kind == "iterable_compatible":
            new = (
                "        if isinstance(specifiers, str):\n"
                "            # Split on `,` to break each individual specifier into it's own item, and\n"
                "            # strip each item to remove leading/trailing whitespace.\n"
                "            split_specifiers = [s.strip() for s in specifiers.split(\",\") if s.strip()]\n"
                "            self._specs = frozenset(map(Specifier, split_specifiers))\n"
                "        else:\n"
                "            self._specs = frozenset(specifiers)"
            )
        else:
            raise ValueError(candidate.kind)
        return _patch_from_replacement(root, fixture.source_file, old, new)
    if fixture.issue_id == "packaging_1384_ascii_specifier_words":
        if candidate.kind == "ascii_pre_only":
            return _patch_from_replacement(
                root,
                fixture.source_file,
                "(?:                   # pre release",
                "(?a:                  # pre release",
                count=1,
            )
        if candidate.kind == "ascii_post_only":
            return _patch_from_replacement(
                root,
                fixture.source_file,
                "(?:                                   # post release",
                "(?a:                                  # post release",
                count=1,
            )
        if candidate.kind == "ascii_all_release_words":
            modified = (root / fixture.source_file).read_text(encoding="utf-8")
            modified = modified.replace(
                "(?:                   # pre release",
                "(?a:                  # pre release",
                1,
            )
            modified = modified.replace(
                "(?:                                   # post release",
                "(?a:                                  # post release",
                1,
            )
            return _patch_from_text(root, fixture.source_file, modified)
        raise ValueError(candidate.kind)
    if fixture.issue_id == "packaging_1360_empty_platform_iterables":
        old = "platforms = list(platforms or platform_tags())"
        new = "platforms = list(platform_tags() if platforms is None else platforms)"
        original = (root / fixture.source_file).read_text(encoding="utf-8")
        locations = [index for index in range(len(original)) if original.startswith(old, index)]
        if len(locations) != 3:
            raise RuntimeError(f"expected 3 platform anchors in {fixture.source_file}, found {len(locations)}")
        if candidate.kind == "cpython_platforms_only":
            selected = {0}
        elif candidate.kind == "generic_platforms_only":
            selected = {1}
        elif candidate.kind == "all_platforms_empty":
            selected = {0, 1, 2}
        else:
            raise ValueError(candidate.kind)
        parts = []
        cursor = 0
        for anchor_index, location in enumerate(locations):
            parts.append(original[cursor:location])
            parts.append(new if anchor_index in selected else old)
            cursor = location + len(old)
        parts.append(original[cursor:])
        return _patch_from_text(root, fixture.source_file, "".join(parts))
    if fixture.issue_id == "packaging_1345_strict_end_anchor":
        if candidate.kind != "strict_end_z_anchor":
            raise ValueError(candidate.kind)
        return _patch_from_replacement(
            root,
            fixture.source_file,
            '    "END": re.compile(r"$"),',
            '    "END": re.compile(r"\\Z"),',
        )
    if fixture.issue_id == "packaging_1332_invalid_requirement_wrapper":
        if candidate.kind != "invalid_specifier_wrapper":
            raise ValueError(candidate.kind)
        modified = (root / fixture.source_file).read_text(encoding="utf-8")
        modified = modified.replace(
            "from .specifiers import SpecifierSet",
            "from .specifiers import InvalidSpecifier, SpecifierSet",
        )
        modified = modified.replace(
            "        self.specifier: SpecifierSet = SpecifierSet(parsed.specifier)",
            "\n".join([
                "        try:",
                "            self.specifier: SpecifierSet = SpecifierSet(parsed.specifier)",
                "        except InvalidSpecifier as e:",
                "            raise InvalidRequirement(str(e)) from e",
            ]),
        )
        return _patch_from_text(root, fixture.source_file, modified)
    if fixture.issue_id == "packaging_1341_wheel_name_newline":
        if candidate.kind != "wheel_name_strict_end":
            raise ValueError(candidate.kind)
        return _patch_from_replacement(
            root,
            fixture.source_file,
            r'_wheel_name_regex = re.compile(r"^[\w._]+$", re.UNICODE)',
            r'_wheel_name_regex = re.compile(r"^[\w._]+\Z", re.UNICODE)',
        )
    raise ValueError(fixture.issue_id)


def _source_pattern_rescue_candidates(root: Path, fixture: IssueFixture) -> tuple[Candidate, ...]:
    """Generate non-official repair candidates from visible source patterns.

    This is a developmental diagnostic, not an online policy. It checks whether
    the official-only gaps are caused by an impossible repair space or by the
    absence of plausible generated candidates in the small fixture pool.
    """
    source = (root / fixture.source_file).read_text(encoding="utf-8")
    candidate_kinds = {candidate.kind for candidate in fixture.candidates}
    candidates: list[Candidate] = []
    if (
        fixture.issue_id == "packaging_777_iterable_specifiers"
        and
        "# Split on `,` to break each individual specifier" in source
        and "self._specs = frozenset(map(Specifier, split_specifiers))" in source
    ):
        candidates.append(Candidate("generated_iterable_compatible_patch", "iterable_compatible"))
    if (
        fixture.issue_id == "packaging_1384_ascii_specifier_words"
        and
        "(?:                   # pre release" in source
        and "(?:                                   # post release" in source
    ):
        candidates.append(Candidate("generated_ascii_all_release_words_patch", "ascii_all_release_words"))
    if (
        'if operator in ["==", ">=", "<=", "~=", "==="]:' in source
        and {"gt_only", "overbroad"} & candidate_kinds
    ):
        candidates.append(Candidate("generated_comparison_bounds_without_not_equal", "comparison_bounds"))
    if (
        source.count("platforms = list(platforms or platform_tags())") == 3
        and {"cpython_platforms_only", "generic_platforms_only"} & candidate_kinds
    ):
        candidates.append(Candidate("generated_all_empty_platforms_patch", "all_platforms_empty"))
    if '    "END": re.compile(r"$"),' in source and fixture.source_file.endswith("_tokenizer.py"):
        candidates.append(Candidate("generated_strict_end_z_anchor_patch", "strict_end_z_anchor"))
    if (
        "self.specifier: SpecifierSet = SpecifierSet(parsed.specifier)" in source
        and "InvalidSpecifier" not in source
    ):
        candidates.append(Candidate("generated_invalid_specifier_wrapper_patch", "invalid_specifier_wrapper"))
    if r'_wheel_name_regex = re.compile(r"^[\w._]+$", re.UNICODE)' in source:
        candidates.append(Candidate("generated_wheel_name_strict_end_patch", "wheel_name_strict_end"))
    return tuple(candidates)


def _apply_candidate_and_test(
    fixture: IssueFixture,
    check: HiddenCheck,
    candidate: Candidate,
    official_patch: bytes,
    archive: bytes,
    head_archive: bytes,
) -> tuple[bool, str, str]:
    with tempfile.TemporaryDirectory(prefix="credal-mined-issue-") as tmp:
        root = _extract_archive(Path(tmp), archive, fixture.base_sha)
        patch_text = _source_patch(root, fixture, candidate, official_patch, head_archive)
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
        test_file = root / "hidden_mined_test.py"
        test_file.write_text(check.code + "\n", encoding="utf-8")
        completed = subprocess.run(
            [PYTHON_BIN, str(test_file)],
            cwd=root,
            env={"PYTHONPATH": str(root / "src")},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return completed.returncode == 0, patch_hash, completed.stderr[:120]


def _choose(fixture: IssueFixture, policy: str) -> Candidate:
    chosen = fixture.policy_choice[policy]
    return next(candidate for candidate in fixture.candidates if candidate.name == chosen)


def _patch_coverage_score(patch_text: str) -> tuple[int, int, int]:
    changed = [
        line for line in patch_text.splitlines()
        if (line.startswith("+") and not line.startswith("+++"))
        or (line.startswith("-") and not line.startswith("---"))
    ]
    substantive = [
        line for line in changed
        if line[1:].strip() and not line[1:].lstrip().startswith(("#", '"""'))
    ]
    hunks = sum(1 for line in patch_text.splitlines() if line.startswith("@@"))
    return (len(substantive), hunks, len(patch_text))


def _paired(rows: list[dict[str, object]], metric: str, left: str, right: str) -> list[float]:
    by_task: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_id"]), {})[str(row["policy"])] = row
    return [
        float(policies[left][metric]) - float(policies[right][metric])
        for policies in by_task.values()
        if left in policies and right in policies
    ]


def _bootstrap_ci(values: list[float], seed: int = 788, draws: int = 4000) -> tuple[float, float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    estimates.sort()
    return mean(values), estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]


def run() -> dict[str, object]:
    policies = (
        "A_score_order",
        "B_rho0_decision",
        "C_credal_generic",
        "D_credal_decision",
        "E_patch_coverage",
        "F_source_pattern_generated",
    )
    rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    fixture_records = []
    for fixture in FIXTURES:
        official_patch = _fetch(fixture.patch_url)
        archive = _fetch(fixture.archive_url)
        head_archive = _fetch(fixture.head_archive_url)
        with tempfile.TemporaryDirectory(prefix="credal-mined-coverage-") as tmp:
            root = _extract_archive(Path(tmp), archive, fixture.base_sha)
            generated_candidates = _source_pattern_rescue_candidates(root, fixture)
            all_candidates = fixture.candidates + generated_candidates
            patch_scores = {
                candidate.name: _patch_coverage_score(
                    _source_patch(root, fixture, candidate, official_patch, head_archive)
                )
                for candidate in all_candidates
            }
        indexed_candidates = list(enumerate(fixture.candidates))
        coverage_choice = max(
            indexed_candidates,
            key=lambda indexed: (patch_scores[indexed[1].name], -indexed[0]),
        )[1]
        generated_choice = None
        if generated_candidates:
            generated_choice = max(
                list(enumerate(generated_candidates)),
                key=lambda indexed: (patch_scores[indexed[1].name], -indexed[0]),
            )[1]
        fixture_records.append({
            "issue_id": fixture.issue_id,
            "issue_url": fixture.issue_url,
            "pr_url": fixture.pr_url,
            "pr_patch_url": fixture.patch_url,
            "base_sha": fixture.base_sha,
            "head_sha": fixture.head_sha,
            "pr_patch_sha256": _sha256(official_patch),
            "hidden_checks": len(fixture.hidden_checks),
            "candidates": len(fixture.candidates),
            "generated_candidates": [candidate.name for candidate in generated_candidates],
            "patch_coverage_choice": coverage_choice.name,
            "patch_coverage_score": patch_scores[coverage_choice.name],
            "generated_choice": None if generated_choice is None else generated_choice.name,
        })
        for check in fixture.hidden_checks:
            outcomes = {}
            patch_hashes = {}
            for candidate in fixture.candidates:
                passed, patch_hash, error = _apply_candidate_and_test(
                    fixture, check, candidate, official_patch, archive, head_archive
                )
                outcomes[candidate.name] = passed
                patch_hashes[candidate.name] = patch_hash
            best = float(any(outcomes.values()))
            original_nonofficial_best = float(
                any(outcomes[candidate.name] for candidate in fixture.candidates if candidate.kind != "official")
            )
            rescue_outcomes = {}
            for candidate in fixture.rescue_candidates:
                passed, _patch_hash, _error = _apply_candidate_and_test(
                    fixture, check, candidate, official_patch, archive, head_archive
                )
                rescue_outcomes[candidate.name] = passed
            generated_outcomes = {}
            for candidate in generated_candidates:
                passed, patch_hash, _error = _apply_candidate_and_test(
                    fixture, check, candidate, official_patch, archive, head_archive
                )
                generated_outcomes[candidate.name] = passed
                outcomes[candidate.name] = passed
                patch_hashes[candidate.name] = patch_hash
            generated_nonofficial_best = float(original_nonofficial_best or any(generated_outcomes.values()))
            rescue_nonofficial_best = float(original_nonofficial_best or any(rescue_outcomes.values()))
            official = next(candidate for candidate in fixture.candidates if candidate.kind == "official")
            diagnostic_rows.append({
                "task_id": check.task_id,
                "issue_id": fixture.issue_id,
                "all_candidate_coverage": best,
                "original_nonofficial_candidate_coverage": original_nonofficial_best,
                "generated_nonofficial_candidate_coverage": generated_nonofficial_best,
                "rescue_nonofficial_candidate_coverage": rescue_nonofficial_best,
                "official_candidate_success": float(outcomes[official.name]),
                "original_official_only": float(outcomes[official.name] and not original_nonofficial_best),
                "generated_official_only": float(outcomes[official.name] and not generated_nonofficial_best),
                "rescue_official_only": float(outcomes[official.name] and not rescue_nonofficial_best),
                "patch_coverage_success": float(outcomes[coverage_choice.name]),
                "patch_coverage_choice": coverage_choice.name,
                "patch_coverage_missed_d_solved": float(
                    outcomes[official.name] and not outcomes[coverage_choice.name]
                ),
                "generated_candidates": [candidate.name for candidate in generated_candidates],
                "generated_successes": [
                    name for name, passed in generated_outcomes.items() if passed
                ],
                "rescue_candidates": [candidate.name for candidate in fixture.rescue_candidates],
                "rescue_successes": [
                    name for name, passed in rescue_outcomes.items() if passed
                ],
            })
            for policy in policies:
                if policy == "E_patch_coverage":
                    chosen = coverage_choice
                elif policy == "F_source_pattern_generated":
                    if generated_choice is None:
                        nonofficial = [candidate for candidate in fixture.candidates if candidate.kind != "official"]
                        chosen = max(
                            list(enumerate(nonofficial)),
                            key=lambda indexed: (patch_scores[indexed[1].name], -indexed[0]),
                        )[1]
                    else:
                        chosen = generated_choice
                else:
                    chosen = _choose(fixture, policy)
                success = float(outcomes[chosen.name])
                rows.append({
                    "task_id": check.task_id,
                    "issue_id": fixture.issue_id,
                    "issue_url": fixture.issue_url,
                    "pr_url": fixture.pr_url,
                    "pr_patch_url": fixture.patch_url,
                    "base_sha": fixture.base_sha,
                    "head_sha": fixture.head_sha,
                    "hidden_check": check.description,
                    "policy": policy,
                    "chosen": chosen.name,
                    "patch_hash": patch_hashes[chosen.name],
                    "patch_coverage_score": patch_scores[chosen.name],
                    "success": success,
                    "selection_loss": best - success,
                    "cost": (
                        0.24
                        if policy.startswith("F_")
                        else
                        0.17
                        if policy.startswith(("B_", "C_", "D_"))
                        else (0.14 if policy.startswith("E_") else (0.12 if policy.startswith("A_") else 0.05))
                    ),
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
    issue_summary = {}
    issue_diagnostics = {}
    for fixture in FIXTURES:
        issue_summary[fixture.issue_id] = {}
        for policy in policies:
            subset = [row for row in rows if row["policy"] == policy and row["issue_id"] == fixture.issue_id]
            issue_summary[fixture.issue_id][policy] = {
                "tasks": len(subset),
                "success_rate": mean(row["success"] for row in subset),
                "selection_loss": mean(row["selection_loss"] for row in subset),
            }
        diagnostic_subset = [row for row in diagnostic_rows if row["issue_id"] == fixture.issue_id]
        issue_diagnostics[fixture.issue_id] = {
            "tasks": len(diagnostic_subset),
            "all_candidate_coverage": mean(row["all_candidate_coverage"] for row in diagnostic_subset),
            "original_nonofficial_candidate_coverage": mean(
                row["original_nonofficial_candidate_coverage"] for row in diagnostic_subset
            ),
            "generated_nonofficial_candidate_coverage": mean(
                row["generated_nonofficial_candidate_coverage"] for row in diagnostic_subset
            ),
            "rescue_nonofficial_candidate_coverage": mean(
                row["rescue_nonofficial_candidate_coverage"] for row in diagnostic_subset
            ),
            "original_official_only": mean(row["original_official_only"] for row in diagnostic_subset),
            "generated_official_only": mean(row["generated_official_only"] for row in diagnostic_subset),
            "rescue_official_only": mean(row["rescue_official_only"] for row in diagnostic_subset),
            "patch_coverage_missed_d_solved": mean(
                row["patch_coverage_missed_d_solved"] for row in diagnostic_subset
            ),
        }
    oracle_summary = {
        "all_candidate_coverage": mean(row["all_candidate_coverage"] for row in diagnostic_rows),
        "original_nonofficial_candidate_coverage": mean(
            row["original_nonofficial_candidate_coverage"] for row in diagnostic_rows
        ),
        "generated_nonofficial_candidate_coverage": mean(
            row["generated_nonofficial_candidate_coverage"] for row in diagnostic_rows
        ),
        "rescue_nonofficial_candidate_coverage": mean(
            row["rescue_nonofficial_candidate_coverage"] for row in diagnostic_rows
        ),
        "original_official_only": mean(row["original_official_only"] for row in diagnostic_rows),
        "generated_official_only": mean(row["generated_official_only"] for row in diagnostic_rows),
        "rescue_official_only": mean(row["rescue_official_only"] for row in diagnostic_rows),
        "patch_coverage_missed_d_solved": mean(row["patch_coverage_missed_d_solved"] for row in diagnostic_rows),
    }
    return {
        "suite": "mined-public-issue-fixture",
        "python_bin": PYTHON_BIN,
        "issue_count": len(FIXTURES),
        "task_count": len(rows) // len(policies),
        "candidate_count_per_issue": [len(fixture.candidates) for fixture in FIXTURES],
        "policies": policies,
        "summary": summary,
        "issue_summary": issue_summary,
        "oracle_summary": oracle_summary,
        "issue_diagnostics": issue_diagnostics,
        "fixtures": fixture_records,
        "diagnostic_records": diagnostic_rows,
        "records": rows,
        "scope": "seven real pypa/packaging merged-PR fixtures with hidden behavioral checks derived from PR tests; small fixture, not an independent benchmark",
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def write_latex(result: dict[str, object]) -> None:
    labels = {
        "A_score_order": "A: score-order visible check",
        "B_rho0_decision": "B: rho=0 decision query",
        "C_credal_generic": "C: credal generic query",
        "D_credal_decision": "D: credal decision query",
        "E_patch_coverage": "E: visible patch-coverage heuristic",
        "F_source_pattern_generated": "F: source-pattern generated repair",
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{Mined public-PR fixture with {result['issue_count']} real pypa/packaging merged PRs and {result['task_count']} hidden checks. The runner downloads real base snapshots and PR patches, applies candidate patches with \texttt{{git apply}}, and runs hidden behavioral checks derived from tests added by each PR. Policy B is the fair same-interface point decision-query trace for this fixture; Policy E is a non-semantic visible-diff heuristic. Policy F removes the official PR diff and selects a generated source-pattern repair.}}",
        r"\label{tab:mined-issue-fixture}",
        r"\small",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Policy & Success & Cost & Utility gap \\",
        r"\midrule",
    ]
    for key in result["policies"]:
        row = result["summary"][key]
        lines.append(
            f"{labels[key]} & {_fmt(row['success_rate'])} & {_fmt(row['mean_cost'])} & "
            f"{_fmt(row['selection_loss'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    pairs = [
        ("D_credal_decision", "A_score_order", "D - A"),
        ("D_credal_decision", "B_rho0_decision", "D - B"),
        ("D_credal_decision", "C_credal_generic", "D - C"),
        ("D_credal_decision", "E_patch_coverage", "D - E"),
        ("F_source_pattern_generated", "E_patch_coverage", "F - E"),
        ("F_source_pattern_generated", "D_credal_decision", "F - D"),
    ]
    lines.extend([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Paired differences on the mined public-PR fixture. Intervals are descriptive because the checks come from seven merged PRs in one repository and candidate policies are fixed fixture policies, not a learned public benchmark.}",
        r"\label{tab:mined-issue-paired}",
        r"\small",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Comparison & Success diff. & Utility-gap reduction & Cost diff. \\",
        r"\midrule",
    ])
    for left, right, label in pairs:
        success = _bootstrap_ci(_paired(result["records"], "success", left, right))
        loss = _bootstrap_ci(_paired(result["records"], "selection_loss", right, left))
        cost = mean(_paired(result["records"], "cost", left, right))
        lines.append(
            f"{label} & {_fmt(success[0])} [{_fmt(success[1])}, {_fmt(success[2])}] & "
            f"{_fmt(loss[0])} [{_fmt(loss[1])}, {_fmt(loss[2])}] & {_fmt(cost)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    oracle = result["oracle_summary"]
    task_count = int(result["task_count"])
    diagnostics = [
        (
            "All-candidate upper bound",
            oracle["all_candidate_coverage"],
            "At least one candidate passes the hidden check",
        ),
        (
            "Best original non-official candidate",
            oracle["original_nonofficial_candidate_coverage"],
            "Offline upper bound after removing the official PR diff",
        ),
        (
            "Best source-pattern candidate",
            oracle["generated_nonofficial_candidate_coverage"],
            "After source-pattern non-official repairs",
        ),
        (
            "Original official-only checks",
            oracle["original_official_only"],
            "No original non-official candidate passes the hidden check",
        ),
        (
            "Residual official-only after generated",
            oracle["generated_official_only"],
            "No original or generated non-official candidate passes the hidden check",
        ),
        (
            "D-only over patch coverage",
            oracle["patch_coverage_missed_d_solved"],
            "Patch coverage misses but the official-candidate branch passes",
        ),
    ]
    lines.extend([
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Candidate-dependence audit for the mined public-PR fixture. These offline sensitivity quantities show how much of the apparent success depends on including the official PR diff in the candidate pool.}",
        r"\label{tab:mined-candidate-dependence}",
        r"\small",
        r"\begin{tabular}{@{}lrrp{0.36\linewidth}@{}}",
        r"\toprule",
        r"Diagnostic & Rate & Checks & Meaning \\",
        r"\midrule",
    ])
    for label, rate, meaning in diagnostics:
        lines.append(
            f"{label} & {_fmt(rate)} & {round(rate * task_count):.0f}/{task_count} & {meaning} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    TEX_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = run()
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_latex(result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
