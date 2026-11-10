#!/usr/bin/env python3
"""Standalone deterministic parser for coding-agent trajectory operations."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from copy import deepcopy
import math
import posixpath
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

PARSER_ID = "behavioral-trajectory-parser"
CODEBOOK_VERSION = "behavioral-trajectory-codebook"
PRIMARY_OPERATIONS = {
    "EXPLORE_REPOSITORY", "SEARCH_REPOSITORY", "READ_FILE", "MUTATE_FILE",
    "RUN_FORMAL_TEST", "INSPECT_WORKSPACE_CHANGES", "PACKAGE_OR_BUILD",
    "OTHER_OPERATION", "NO_REPOSITORY_OPERATION", "UNKNOWN",
}
SEMANTIC_OPERATIONS = PRIMARY_OPERATIONS - {"NO_REPOSITORY_OPERATION", "UNKNOWN"}
TEST_DIRECTORY_RE = re.compile(r"(?:^|/)(?:tests?|__tests__|spec|specs)(?:/|$)", re.I)
TEST_FILENAME_RE = re.compile(
    r"(?:^|/)(?:test_[^/]+|[^/]+_test|[^/]+_spec|[^/]+\.test|[^/]+\.spec|conftest)\.[^/]+$",
    re.I,
)
# Compatibility export: exactly the union of the two repository-agnostic rules.
TEST_PATH_RE = re.compile(
    rf"(?:{TEST_DIRECTORY_RE.pattern})|(?:{TEST_FILENAME_RE.pattern})",
    re.I,
)
COMMAND_BLOCK_RE = re.compile(r"```command\s*\n(.*?)\n```", re.I | re.S)
VIEWER_RE = re.compile(r"^(?:goto|scroll_up|scroll_down)(?:\s|$)", re.I)
FILE_HEADER_RE = re.compile(r"^\[File:\s*(.+?)\s*\([^\n]*lines total\)\]", re.M)
NUMBERED_LINE_RE = re.compile(r"^\s*\d+:", re.M)
HEREDOC_RE = re.compile(r"<<-?\s*['\"]?[A-Za-z_]\w*")
HEREDOC_BOUNDARY_RE = re.compile(r"<<(?P<strip>-?)\s*(?P<quote>['\"]?)(?P<delimiter>[A-Za-z_]\w*)(?P=quote)")
REDIRECT_RE = re.compile(r"(?<![0-9>&-])>>?\s*(?P<dst>[^\s;&|<>]+)")
EXTERNAL_PREFIXES = ("/dev/", "/tmp/", "/var/tmp/", "/proc/", "/sys/")

TEST_PATTERNS = [re.compile(p, re.I) for p in [
    r"^(?:python3?\s+-m\s+)?(?:pytest|py\.test)(?:\s|$)",
    r"^python3?\s+-m\s+(?:unittest|nose2|nose)(?:\s|$)",
    r"^python3?\s+manage\.py\s+test(?:\s|$)",
    r"^(?:tox|nox)(?:\s|$)", r"^cargo\s+(?:test|nextest\s+run)(?:\s|$)",
    r"^go\s+test(?:\s|$)", r"^gotestsum(?:\s|$)",
    r"^(?:npm|yarn|pnpm|bun)\s+(?:run\s+)?(?:test|test:[\w:-]+|[\w-]*:test)(?:\s|$)",
    r"^(?:jest|vitest|mocha|ava|karma|playwright\s+test|cypress\s+run)(?:\s|$)",
    r"^(?:mvn|mvnw|gradle|gradlew)\b(?=.*\b(?:test|tests|check|verify)\b)",
    r"^(?:bazel|buck2?)\s+test(?:\s|$)", r"^make\s+(?:test|tests|check)(?:\s|$)",
    r"^ctest(?:\s|$)", r"^(?:bundle\s+exec\s+)?rspec(?:\s|$)", r"^dotnet\s+test(?:\s|$)",
]]
STRONG_FAILURE_RE = re.compile(
    r"\b[1-9]\d*\s+failed\b|\b[1-9]\d*\s+errors?\b|^FAILED\b|"
    r"^[ \t]*(?:---\s+|===\s+)?FAIL\b|test result:\s*FAILED|"
    r"\bFailures?:\s*[1-9]\d*\b|\bErrors?:\s*[1-9]\d*\b|BUILD FAIL(?:URE|ED)|"
    r"(?:build|module) cache (?:is required, but could not be located|not found)|"
    r"=+\s*failures?\s*=+|MojoFailureException|MavenExecutionException|"
    r"DependencyResolutionException|Compilation failure|No tests were executed|"
    r"Could not find the selected project in the reactor|ERR_PNPM|ELIFECYCLE|"
    r"Command failed with exit code",
    re.I | re.M,
)
CONTEXTUAL_FAILURE_RE = re.compile(
    r"ImportError while importing test module|"
    r"^[ \t]*E[ \t]+(?:ModuleNotFoundError|ImportError|AssertionError)\b|"
    r"Traceback \(most recent call last\):[\s\S]{0,4000}?"
    r"(?:AssertionError(?:\s*:\s*[^\n]+)?|ModuleNotFoundError(?:\s*:\s*[^\n]+|\s+No module named[^\n]*)|ImportError\s*:\s*[^\n]+)",
    re.I | re.M,
)
LOOSE_FAILURE_RE = re.compile(
    r"ModuleNotFoundError|ImportError while loading|^npm error\b|^\[ERROR\]|"
    r"assertionerror|errors? during collection",
    re.I | re.M,
)
POST_SUMMARY_FAILURE_RE = re.compile(
    r"^[^\n]*(?:Segmentation fault(?:\s*\(core dumped\))?|Aborted\s*\(core dumped\))\s*$|"
    r"^Fatal Python error:|^INTERNALERROR>|^panic:\s|^fatal error:\s",
    re.I | re.M,
)
# Compiled pattern exported for callers.
FAILURE_RE = re.compile(
    "|".join((STRONG_FAILURE_RE.pattern, CONTEXTUAL_FAILURE_RE.pattern, LOOSE_FAILURE_RE.pattern, POST_SUMMARY_FAILURE_RE.pattern)),
    re.I | re.M,
)
PASS_RE = re.compile(
    r"\b\d+\s+passed\b|\b\d+\s+passing\b|test result:\s*ok|^ok\s+\S+|^OK\s*$|"
    r"^[ \t]*(?:---\s+)?PASS\b|\btests? passed\b|BUILD SUCCESS(?:FUL)?|SUCCESS:\s*Executed|"
    r"Tests run:.*Failures?:\s*0\b.*Errors?:\s*0\b|\[no test files\]",
    re.I | re.M,
)

FLAGS_WITH_VALUE = {
    "head": {"-n", "-c", "--lines", "--bytes"}, "tail": {"-n", "-c", "--lines", "--bytes"},
    "sed": {"-e", "-f", "--expression", "--file"},
    "grep": {"-e", "-f", "-m", "-A", "-B", "-C", "--include", "--exclude", "--regexp"},
    "rg": {"-e", "-g", "-m", "-A", "-B", "-C", "-t", "--glob", "--type", "--regexp"},
    "ag": {"-A", "-B", "-C", "--ignore"},
    "find": {"-name", "-iname", "-path", "-type", "-maxdepth", "-mindepth", "-not"},
    "fd": {"-e", "-t", "-d", "--extension", "--type"},
    "search_file": set(), "search_dir": set(), "find_file": set(),
    "wc": set(), "nl": set(), "cat": set(), "less": set(), "open": set(),
}
PATTERN_FIRST = {"grep", "rg", "ag", "sed", "search_file", "search_dir", "find_file"}
TOOL_OPERATIONS = {
    "read": "READ_FILE", "open": "READ_FILE", "view": "READ_FILE",
    "edit": "MUTATE_FILE", "replace": "MUTATE_FILE", "multiedit": "MUTATE_FILE",
    "write": "MUTATE_FILE", "str_replace": "MUTATE_FILE", "str_replace_editor": "MUTATE_FILE",
    "apply_patch": "MUTATE_FILE", "create": "MUTATE_FILE", "writefile": "MUTATE_FILE",
    "grep": "SEARCH_REPOSITORY", "glob": "SEARCH_REPOSITORY", "search": "SEARCH_REPOSITORY",
    "search_file": "SEARCH_REPOSITORY", "search_dir": "SEARCH_REPOSITORY", "find_file": "SEARCH_REPOSITORY",
    "submit": "NO_REPOSITORY_OPERATION", "finish": "NO_REPOSITORY_OPERATION",
}
TOOL_MUTATIONS = {
    "edit": "EDIT", "replace": "EDIT", "multiedit": "EDIT", "str_replace": "EDIT",
    "str_replace_editor": "EDIT", "write": "EDIT", "create": "CREATE", "writefile": "CREATE",
}
PATCH_HEADER_RE = re.compile(r"^\*\*\*\s+(?P<kind>Add|Update|Delete)\s+File:\s*(?P<path>.+?)\s*$", re.M | re.I)
DIFF_HEADER_RE = re.compile(r"^\+\+\+\s+(?:b/)?(?P<path>[^\s]+)\s*$", re.M)
ARGUMENT_PATH_KEYS = (
    "file_path", "path", "filename", "file", "FILE", "FILENAME", "notebook_path",
    "target_file", "absolute_path",
)


def split_top_level(text: str) -> list[str]:
    """Split shell separators outside quotes while preserving left-to-right order."""
    out: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    i = 0
    while i < len(text):
        ch = text[i]
        if escaped:
            buf.append(ch); escaped = False; i += 1; continue
        if ch == "\\" and quote != "'":
            buf.append(ch); escaped = True; i += 1; continue
        if quote:
            buf.append(ch)
            if ch == quote: quote = None
            i += 1; continue
        if ch == "#" and (i == 0 or text[i - 1].isspace()):
            end = text.find("\n", i)
            if end < 0: break
            out.append("".join(buf)); buf = []; i = end + 1; continue
        if ch in "'\"":
            quote = ch; buf.append(ch); i += 1; continue
        if text.startswith("&&", i) or text.startswith("||", i):
            out.append("".join(buf)); buf = []; i += 2; continue
        if ch in ";|\n":
            out.append("".join(buf)); buf = []; i += 1; continue
        buf.append(ch); i += 1
    out.append("".join(buf))
    return [x.strip() for x in out if x.strip()]


def open_shell_quote(text: str) -> str | None:
    quote: str | None = None
    escaped = False
    i = 0
    while i < len(text):
        ch = text[i]
        if escaped:
            escaped = False; i += 1; continue
        if ch == "\\" and quote != "'":
            escaped = True; i += 1; continue
        if quote:
            if ch == quote: quote = None
            i += 1; continue
        if ch == "#" and (i == 0 or text[i - 1].isspace()):
            end = text.find("\n", i)
            if end < 0: break
            i = end + 1; continue
        if ch in "'\"": quote = ch
        i += 1
    return quote


def strip_wrappers(segment: str) -> str:
    text = segment.strip()
    text = re.sub(r"^(?:env\s+)?(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)+", "", text)
    text = re.sub(r"^(?:timeout\s+\S+\s+|nohup\s+|xvfb-run\s+(?:-a\s+)?)+", "", text, flags=re.I)
    text = re.sub(r"^(?:uv|poetry|pipenv|pdm|rye|hatch)\s+run\s+", "", text, flags=re.I)
    text = re.sub(r"^(?:npx|pnpm\s+dlx|yarn\s+dlx|bunx)\s+(?:--yes\s+|-y\s+)?", "", text, flags=re.I)
    text = re.sub(r"^\S*/(?=(?:pytest|py\.test|jest|vitest|mocha|ava|rspec|tox|nox|gradlew|ctest)\b)", "", text)
    text = re.sub(r"^\./(?=gradlew\b)", "", text)
    return text.strip()


def join_cwd(cwd: str | None, target: str) -> str | None:
    target = target.replace("\\", "/")
    if target.startswith("/"): return posixpath.normpath(target)
    return posixpath.normpath(posixpath.join(cwd, target)) if cwd else None


def unwrap_leading_shell_wrapper(command: str) -> tuple[str, bool]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command, False
    if len(tokens) >= 3 and posixpath.basename(tokens[0]).lower() in {"sh", "bash"} and tokens[1] in {"-c", "-lc"}:
        return tokens[2], True
    return command, False


def segments_with_cwd(command: str, base_cwd: str | None) -> list[dict[str, Any]]:
    """Split commands while keeping heredoc bodies attached to their opening segment."""
    text, shell_wrapper_unwrapped = unwrap_leading_shell_wrapper(command.strip())
    if not text:
        return []
    lines = text.splitlines()
    raw: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        logical_line = lines[i]
        logical_end = i
        while open_shell_quote(logical_line) and logical_end + 1 < len(lines):
            logical_end += 1
            logical_line += "\n" + lines[logical_end]
        parts = split_top_level(logical_line)
        search_from = logical_end + 1
        next_line = logical_end + 1
        for part in parts:
            marker = HEREDOC_BOUNDARY_RE.search(part)
            if not marker:
                raw.append({"head": part, "text": part, "heredoc_unterminated": False})
                continue
            delimiter = marker.group("delimiter")
            strip_tabs = marker.group("strip") == "-"
            end = None
            for j in range(search_from, len(lines)):
                candidate = lines[j].lstrip("\t") if strip_tabs else lines[j]
                if candidate == delimiter:
                    end = j
                    break
            if end is None:
                raw.append({"head": part, "text": "\n".join([part, *lines[search_from:]]), "heredoc_unterminated": True})
                search_from = len(lines)
                next_line = len(lines)
            else:
                raw.append({"head": part, "text": "\n".join([part, *lines[search_from:end + 1]]), "heredoc_unterminated": False})
                search_from = end + 1
                next_line = max(next_line, end + 1)
        i = next_line
    result: list[dict[str, Any]] = []
    cwd = base_cwd
    for item in raw:
        stripped = strip_wrappers(item["head"])
        cd = re.match(r"^cd\s+(?:--\s+)?([^\s;&|]+)\s*$", stripped, re.I)
        if cd:
            cwd = join_cwd(cwd, cd.group(1).strip("'\""))
            continue
        if stripped:
            result.append({"head": stripped.lower(), "head_raw": stripped, "text": item["text"], "cwd": cwd,
                           "heredoc_unterminated": item["heredoc_unterminated"],
                           "shell_wrapper_unwrapped": shell_wrapper_unwrapped})
    return result


def resolve_path(path: str | None, cwd: str | None, repo_root: str | None) -> str | None:
    if not path or not repo_root: return None
    text = str(path).strip().strip("'\"").replace("\\", "/")
    if (
        not text or text.startswith(("-", "~")) or re.fullmatch(r"\d+(?::\d+)?", text)
        or any(c in text for c in "*?$`")
    ):
        return None
    root = posixpath.normpath(repo_root)
    absolute = posixpath.normpath(text if text.startswith("/") else posixpath.join(cwd or root, text))
    if absolute != root and not absolute.startswith(root.rstrip("/") + "/"): return None
    relative = absolute[len(root.rstrip("/")):].lstrip("/")
    if relative in {"", ".", ".."} or relative.startswith("../") or PurePosixPath(relative).is_absolute(): return None
    return relative


def tokenize(text: str) -> list[str]:
    try: return shlex.split(text)
    except ValueError: return text.split()


def command_operands(head_raw: str) -> tuple[str, list[str]]:
    tokens = tokenize(head_raw)
    if not tokens: return "", []
    name = posixpath.basename(tokens[0]).lower(); flags = FLAGS_WITH_VALUE.get(name, set()); operands = []; i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "--": operands.extend(tokens[i + 1:]); break
        if token.startswith("-") and len(token) > 1:
            i += 2 if token in flags else 1; continue
        operands.append(token); i += 1
    if name in PATTERN_FIRST and operands and not any(x in tokens for x in ("-e", "-f", "--regexp", "--expression")):
        operands = operands[1:]
    return name, operands


def path_operands(head_raw: str) -> list[str]:
    _, operands = command_operands(head_raw)
    marker = re.search(r"<<-?\s*['\"]?([A-Za-z_]\w*)", head_raw)
    delimiter = marker.group(1) if marker else None
    return [x for x in operands if not x.startswith("<<") and x != delimiter]


def read_scope(head: str, arguments: dict[str, Any]) -> str:
    if arguments.get("offset") is not None or arguments.get("limit") is not None: return "BOUNDED_RANGE"
    low = head.lower()
    if re.search(r"^\s*sed\s+.*-n", low) or re.search(r"^\s*(?:head|tail)\b", low): return "BOUNDED_RANGE"
    return "FULL_FILE"


def patch_targets(text: str) -> list[tuple[str, str]]:
    targets = [
        ({"add": "CREATE", "update": "EDIT", "delete": "DELETE"}[m.group("kind").lower()], m.group("path"))
        for m in PATCH_HEADER_RE.finditer(text)
    ]
    if not targets:
        targets = [("EDIT", m.group("path")) for m in DIFF_HEADER_RE.finditer(text) if m.group("path") != "/dev/null"]
    return targets


def argument_paths(arguments: dict[str, Any]) -> list[str]:
    paths = [str(arguments[k]).strip() for k in ARGUMENT_PATH_KEYS if arguments.get(k)]
    for key in ("paths", "file_paths", "files"):
        if isinstance(arguments.get(key), list): paths.extend(str(x).strip() for x in arguments[key] if x)
    return [x for x in paths if not re.fullmatch(r"\d+(?::\d+)?", x)]


def split_redirects(text: str) -> tuple[str, list[str]]:
    """Extract shell file redirects outside quotes; leave fd duplication and code syntax alone."""
    base: list[str] = []
    destinations: list[str] = []
    quote: str | None = None
    escaped = False
    i = 0
    while i < len(text):
        ch = text[i]
        if escaped:
            base.append(ch); escaped = False; i += 1; continue
        if ch == "\\" and quote != "'":
            base.append(ch); escaped = True; i += 1; continue
        if quote:
            base.append(ch)
            if ch == quote: quote = None
            i += 1; continue
        if ch in "'\"":
            quote = ch; base.append(ch); i += 1; continue
        if ch != ">" or (i and text[i - 1] == "=") or (i + 1 < len(text) and text[i + 1] in "=&("):
            base.append(ch); i += 1; continue
        j = i + 2 if i + 1 < len(text) and text[i + 1] == ">" else i + 1
        while j < len(text) and text[j].isspace(): j += 1
        if j >= len(text):
            base.append(ch); i += 1; continue
        start = j
        if text[j] in "'\"":
            destination_quote = text[j]; j += 1
            while j < len(text):
                if text[j] == "\\" and destination_quote != "'" and j + 1 < len(text): j += 2; continue
                if text[j] == destination_quote: j += 1; break
                j += 1
        else:
            while j < len(text) and not text[j].isspace() and text[j] not in ";&|<>": j += 1
        destination = text[start:j]
        if not destination:
            base.append(ch); i += 1; continue
        destinations.append(destination)
        base.append(" ")
        i = j
    return "".join(base), destinations


def inline_python_mutation(text: str) -> tuple[bool, list[str], str]:
    path_patterns = [
        (r"open\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wax+]", "inline_python_open_write"),
        (r"Path\(\s*['\"]([^'\"]+)['\"]\s*\)\.open\(\s*['\"][wax+]", "inline_python_path_open_write"),
        (r"Path\(\s*['\"]([^'\"]+)['\"]\s*\)\.(?:write_text|write_bytes|unlink)\s*\(", "inline_python_path_mutation"),
    ]
    paths: list[str] = []
    rule = ""
    matched = False
    for pattern, rid in path_patterns:
        found = re.findall(pattern, text, re.I | re.S)
        if found:
            matched = True; paths.extend(found); rule = rule or rid
    assignments = dict(re.findall(r"\b([A-Za-z_]\w*)\s*=\s*Path\(\s*['\"]([^'\"]+)['\"]\s*\)", text, re.I | re.S))
    variable_mutations = re.findall(r"\b([A-Za-z_]\w*)\.(?:write_text|write_bytes|unlink)\s*\(", text, re.I | re.S)
    variable_open = re.findall(r"\b([A-Za-z_]\w*)\.open\(\s*['\"][wax+]", text, re.I | re.S)
    for variable in variable_mutations + variable_open:
        if variable in assignments: paths.append(assignments[variable])
    broad = re.search(
        r"\b(?:os\.(?:replace|rename|remove|unlink)|shutil\.(?:copy|copy2|copyfile|move)|json\.dump)\s*\(",
        text, re.I | re.S,
    )
    matched = matched or bool(variable_mutations or variable_open or broad)
    return matched, list(dict.fromkeys(paths)), rule or ("inline_python_variable_path_mutation" if variable_mutations else "inline_python_path_open_write" if variable_open else "inline_python_mutation_api" if broad else "")


def classify_segment(record: dict[str, Any], repo_root: str | None) -> dict[str, Any]:
    base, redirects = split_redirects(record["head"])
    base_raw, _ = split_redirects(record["head_raw"])
    root = posixpath.normpath(repo_root) if repo_root else None
    cwd = posixpath.normpath(record["cwd"]) if record.get("cwd") else root
    cwd_inside_repository = bool(root and cwd and (cwd == root or cwd.startswith(root.rstrip("/") + "/")))
    explicit_external_redirects = [
        raw for raw in redirects if raw.strip("'\"").startswith(EXTERNAL_PREFIXES)
    ]
    dynamic_redirects = [
        raw for raw in redirects
        if raw not in explicit_external_redirects and cwd_inside_repository
        and ("$" in raw or "`" in raw or raw.strip("'\"").startswith("~"))
    ]
    redirect_pairs = [
        (raw, resolve_path(raw, record["cwd"], repo_root))
        for raw in redirects
        if raw not in dynamic_redirects and raw not in explicit_external_redirects
    ]
    repository_redirects = [raw for raw, resolved in redirect_pairs if resolved]
    external_redirect = any(
        raw not in dynamic_redirects and raw not in repository_redirects for raw in redirects
    )
    result = {"operations": ["OTHER_OPERATION"], "mutation": None, "paths": [], "mutation_paths": [],
              "terminal": False, "read_scope": "NOT_APPLICABLE", "external_redirect": external_redirect,
              "cwd": record["cwd"], "rule_ids": ["fallback_other"]}

    def set_base(operation: str, rule_id: str, *, mutation: str | None = None,
                 paths: list[str] | None = None, read_scope_value: str = "NOT_APPLICABLE",
                 terminal: bool = False) -> None:
        result.update(operations=[operation], mutation=mutation, paths=paths or [],
                      mutation_paths=(paths or []) if mutation else [], terminal=terminal,
                      read_scope=read_scope_value, rule_ids=[rule_id])

    if re.match(r"^(?:submit|finish)\b", base):
        set_base("NO_REPOSITORY_OPERATION", "terminal_submit", terminal=True)
    else:
        patches = patch_targets(record["text"])
        if re.match(r"^(?:edit|replace|apply_patch)\b", base) or patches:
            kinds = {k for k, _ in patches} or {"EDIT"}
            paths = [p for _, p in patches] or path_operands(base_raw)
            set_base("MUTATE_FILE", "patch_or_edit", mutation="MIXED" if len(kinds) > 1 else next(iter(kinds)), paths=paths)
        elif re.match(r"^(?:create|touch)\b", base):
            set_base("MUTATE_FILE", "create_command", mutation="CREATE", paths=path_operands(base_raw))
        elif re.match(r"^(?:rm|unlink)\b", base):
            set_base("MUTATE_FILE", "delete_command", mutation="DELETE", paths=path_operands(base_raw))
        elif re.search(r"(?:^|\s)(?:sed|perl)\s+(?:\S+\s+)*-i\b", base):
            tokens = tokenize(base_raw)
            paths = [x for x in tokens[1:] if not x.startswith("-")][1:] or tokens[-1:]
            set_base("MUTATE_FILE", "inplace_edit", mutation="EDIT", paths=paths)
        else:
            inline, paths, rid = inline_python_mutation(record["text"])
            if inline:
                set_base("MUTATE_FILE", rid, mutation="EDIT", paths=paths)
            elif re.match(r"^(?:open|cat|head|tail|less|nl|wc|sed|goto|scroll_up|scroll_down)\b", base):
                name, _ = command_operands(base_raw)
                operands = path_operands(base_raw)
                if name not in {"goto", "scroll_up", "scroll_down"} and not operands:
                    result["rule_ids"] = ["stream_command_without_repository_operand"]
                else:
                    set_base("READ_FILE", "read_command", paths=operands, read_scope_value=read_scope(base_raw, {}))
            elif re.match(r"^(?:rg|grep|find|fd|ag|ack|search_file|search_dir|find_file)\b", base):
                set_base("SEARCH_REPOSITORY", "search_command", paths=command_operands(base_raw)[1])
            elif re.match(r"^(?:ls|tree|pwd|du|stat)\b", base):
                set_base("EXPLORE_REPOSITORY", "explore_command")
            elif re.search(r"^git\b(?:\s+--\S+)*\s+(?:diff|status|show)\b", base):
                set_base("INSPECT_WORKSPACE_CHANGES", "workspace_inspection")
            elif any(pattern.search(base) for pattern in TEST_PATTERNS):
                set_base("RUN_FORMAL_TEST", "formal_test_registry")
            elif re.search(r"\b(?:pip|pip3|uv|poetry|conda|npm|yarn|pnpm|apt-get|cargo|go|mvn|gradle)\s+(?:install|add|build|get|sync)\b|^(?:make|cmake|configure)\b", base):
                set_base("PACKAGE_OR_BUILD", "package_or_build")

    if repository_redirects:
        if result["operations"] == ["OTHER_OPERATION"]:
            result["operations"] = ["MUTATE_FILE"]
            result["rule_ids"] = ["repository_redirect"]
        else:
            result["operations"].append("MUTATE_FILE")
            result["rule_ids"].append("repository_redirect")
        if result["mutation"] not in {None, "EDIT"}:
            result["mutation"] = "MIXED"
        else:
            result["mutation"] = "EDIT"
        result["mutation_paths"].extend(repository_redirects)
    if dynamic_redirects:
        result["rule_ids"].append("unresolved_redirect_destination")
    if record.get("heredoc_unterminated"): 
        result["rule_ids"].append("heredoc_unterminated")
    if record.get("shell_wrapper_unwrapped"):
        result["rule_ids"].append("shell_wrapper_unwrapped")
    return result


def recover_command_from_invalid_json(raw: str) -> str:
    match = re.search(r"['\"]command['\"]\s*:\s*['\"]", raw, re.I)
    if not match: return raw
    out: list[str] = []; i = match.end(); escaped = False
    escapes = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", "\\": "\\", "\"": "\"", "'": "'", "/": "/"}
    while i < len(raw):
        ch = raw[i]
        if escaped:
            if ch == "u" and i + 4 < len(raw) and re.fullmatch(r"[0-9a-fA-F]{4}", raw[i + 1:i + 5]):
                out.append(chr(int(raw[i + 1:i + 5], 16))); i += 5; escaped = False; continue
            out.append(escapes.get(ch, ch)); escaped = False; i += 1; continue
        if ch == "\\": escaped = True; i += 1; continue
        if ch in "'\"" and re.match(r"\s*(?:[,}])", raw[i + 1:]): break
        out.append(ch); i += 1
    return "".join(out)


def parse_call(call: dict[str, Any]) -> tuple[str, dict[str, Any], str, bool]:
    function = call.get("function") or call; tool = str(function.get("name") or call.get("name") or "unknown")
    arguments = function.get("arguments", call.get("arguments", {})); invalid = False
    if isinstance(arguments, str):
        raw_arguments = arguments
        try: arguments = json.loads(arguments)
        except json.JSONDecodeError: arguments, invalid = {"raw": raw_arguments, "command": recover_command_from_invalid_json(raw_arguments)}, True
    if not isinstance(arguments, dict): arguments, invalid = {"value": arguments}, True
    command = str(arguments.get("command") or arguments.get("cmd") or arguments.get("input") or arguments.get("patch") or arguments.get("raw") or "")
    if not command and re.search(r"[\s/;&|]", tool): command, invalid = tool, True
    return tool, arguments, command, invalid


def observation_text(observation: dict[str, Any] | None) -> str:
    if not observation: return ""
    parts = [str(observation["output"])] if observation.get("output") is not None else []
    error = observation.get("error")
    if isinstance(error, dict): parts.extend(str(error[k]) for k in ("output", "partial_output", "message") if error.get(k) is not None)
    elif error is not None: parts.append(str(error))
    return "\n".join(parts)


def observation_exit_code(observation: dict[str, Any] | None) -> int | None:
    if not observation: return None
    for key in ("exit_code", "returncode", "return_code", "status_code"):
        value = observation.get(key)
        if isinstance(value, int) and not isinstance(value, bool): return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()): return int(value)
    error = observation.get("error")
    if isinstance(error, dict) and isinstance(error.get("exit_code"), int): return int(error["exit_code"])
    return None


def index_observations(messages: list[dict[str, Any]]) -> dict[str, Any]:
    index = {"call": collections.defaultdict(list), "response": collections.defaultdict(list), "parent": collections.defaultdict(list)}
    for m in messages:
        if m.get("kind") != "observation": continue
        for source, key in (("call", "tool_call_id"), ("response", "response_id"), ("parent", "parent_id")):
            if m.get(key): index[source][str(m[key])].append(m)
    return index


def pair_observation(
    index: dict[str, Any], call_id: str, action_id: str, *,
    allow_parent_fallback: bool = True,
    consumed_observation_ids: set[int] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Pair one unconsumed observation; parent fallback is caller-gated."""
    consumed = consumed_observation_ids if consumed_observation_ids is not None else set()

    def available(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [item for item in items if id(item) not in consumed]

    matches = available(index["call"].get(call_id, [])) if call_id else []
    saw = bool(matches)
    narrowed = [x for x in matches if action_id and str(x.get("parent_id") or "") == action_id]
    if len(narrowed) == 1: matches = narrowed
    if len(matches) != 1 and call_id:
        matches = available(index["response"].get(call_id, []))
        saw |= bool(matches)
        narrowed = [x for x in matches if action_id and str(x.get("parent_id") or "") == action_id]
        if len(narrowed) == 1: matches = narrowed
    if len(matches) != 1 and allow_parent_fallback:
        children = available(index["parent"].get(action_id, [])) if action_id else []
        saw |= bool(children)
        if len(children) == 1: matches = children
    if len(matches) == 1:
        consumed.add(id(matches[0]))
        return matches[0], "paired"
    return None, "ambiguous" if saw else "missing"


def calls_from_action(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls")
    if isinstance(calls, list) and calls: return calls
    content = str(message.get("content") or ""); blocks = COMMAND_BLOCK_RE.findall(content); command = blocks[-1] if blocks else ""
    if not command and content.strip():
        try:
            value = json.loads(content.rstrip().splitlines()[-1]); command = str(value.get("command") or "") if isinstance(value, dict) else ""
        except json.JSONDecodeError: pass
    return [] if not command else [{"id": f"synthetic-command:{message.get('id', '')}", "function": {"name": "shell", "arguments": {"command": command, "_multiple_command_blocks": len(blocks) > 1}}}]


def classify_test_result_evidence(output: str, exit_code: int | None) -> tuple[str, str, list[str], bool]:
    """Return a conservative event-level verdict and inspectable evidence."""
    text = output or ""
    post_failure = bool(POST_SUMMARY_FAILURE_RE.search(text))
    strong_failure = bool(STRONG_FAILURE_RE.search(text))
    contextual_failure = bool(CONTEXTUAL_FAILURE_RE.search(text))
    strong_pass = bool(PASS_RE.search(text))
    loose_failure = bool(LOOSE_FAILURE_RE.search(text))
    evidence = []
    if post_failure: evidence.append("post_summary_process_failure")
    if strong_failure: evidence.append("strong_failure_summary")
    if contextual_failure: evidence.append("contextual_failure_trace")
    if strong_pass: evidence.append("strong_pass_summary")
    if loose_failure: evidence.append("loose_failure_signal")
    if exit_code == 0: evidence.append("explicit_zero_exit_code")
    elif isinstance(exit_code, int): evidence.append("explicit_nonzero_exit_code")
    decisive_text_failure = post_failure or strong_failure or contextual_failure
    status_conflict = bool(
        (exit_code == 0 and decisive_text_failure)
        or (isinstance(exit_code, int) and exit_code != 0 and strong_pass and not decisive_text_failure)
    )
    pass_with_non_summary_failure = bool(
        strong_pass and not strong_failure and (post_failure or contextual_failure or loose_failure)
    )
    conflict = status_conflict or pass_with_non_summary_failure
    if post_failure: return "failed", "post_summary_process_failure", evidence, conflict
    if strong_failure: return "failed", "strong_failure_summary", evidence, conflict
    if contextual_failure: return "failed", "contextual_failure_trace", evidence, conflict
    if isinstance(exit_code, int) and exit_code != 0: return "failed", "explicit_nonzero_exit_code", evidence, conflict
    if strong_pass: return "passed", "strong_pass_summary", evidence, conflict
    if exit_code == 0: return "passed", "explicit_zero_exit_code", evidence, conflict
    if loose_failure: return "failed", "loose_failure_signal", evidence, conflict
    return "unknown", "no_result_evidence", evidence, False


def classify_test_result(output: str, exit_code: int | None) -> str:
    return classify_test_result_evidence(output, exit_code)[0]


def canonical_signature(command: str, operations: list[str], paths: list[str]) -> str:
    normalized = re.sub(r"\s+", " ", command.strip()).lower()
    return "|".join([",".join(operations), normalized, ",".join(sorted(paths))])


def path_role(path: str) -> tuple[str, str]:
    normalized = posixpath.normpath(path)
    if TEST_FILENAME_RE.search(normalized): return "TEST_FILE", "test_filename"
    if TEST_DIRECTORY_RE.search(normalized): return "TEST_FILE", "generic_test_directory"
    return "NON_TEST_REPOSITORY_FILE", "non_test_default"


def role_of_with_rules(paths: Iterable[str]) -> tuple[str, list[str]]:
    classified = [path_role(path) for path in paths]
    if not classified: return "UNRESOLVED", ["unresolved_no_paths"]
    roles = {role for role, _ in classified}
    rules = list(dict.fromkeys(rule for _, rule in classified))
    if len(roles) > 1: return "MIXED", [*rules, "mixed_target_roles"]
    return next(iter(roles)), rules


def role_of(paths: Iterable[str]) -> str:
    return role_of_with_rules(paths)[0]


def parse_trajectory(
    trajectory: dict[str, Any], rollout_id: str = "rollout", repository_root: str | None = None,
    repository_root_source: str | None = None,
) -> dict[str, Any]:
    messages = trajectory.get("messages") or []
    trajectory_root = trajectory.get("repository_root") or trajectory.get("repo_root")
    root = repository_root or trajectory_root
    if repository_root:
        root_source = repository_root_source or "argument"
    elif trajectory_root:
        root_source = str(trajectory.get("repository_root_source") or "trajectory")
    else:
        root_source = "missing"
    root = posixpath.normpath(str(root)) if root else None
    index = index_observations(messages); events = []; current_file: str | None = None
    consumed_observation_ids: set[int] = set()
    for step, message in enumerate(messages):
        if message.get("kind") != "action": continue
        action_id = str(message.get("id") or "")
        calls = calls_from_action(message)
        for call_index, call in enumerate(calls):
            call_id = str(call.get("id") or f"missing:{step}:{call_index}")
            observation, pairing = pair_observation(
                index, call_id, action_id,
                allow_parent_fallback=len(calls) == 1,
                consumed_observation_ids=consumed_observation_ids,
            )
            tool, arguments, command, invalid = parse_call(call); error = (observation or {}).get("error")
            error_type = str(error.get("type") or "") if isinstance(error, dict) else ("error" if error else "")
            if error_type in {"multiple_commands_extracted", "multiple_tool_calls_extracted"} or arguments.get("_multiple_command_blocks") is True: execution = "rejected_multi_command_attempt"
            elif invalid: execution = "rejected_invalid_arguments_attempt"
            elif pairing == "missing": execution = "observation_missing"
            elif pairing == "ambiguous": execution = "parser_unknown"
            else: execution = "executed"
            output = observation_text(observation); exit_code = observation_exit_code(observation)
            cwd = str((observation or {}).get("current_dir") or root or "") or None
            records = segments_with_cwd(command, cwd); segments = [classify_segment(r, root) for r in records]
            if arguments.get("_multiple_command_blocks") is True:
                segments = [{"operations": ["UNKNOWN"], "mutation": None, "paths": [], "mutation_paths": [],
                             "terminal": False, "read_scope": "NOT_APPLICABLE", "external_redirect": False,
                             "cwd": cwd, "rule_ids": ["multiple_command_blocks_unknown"]}]
                records = [{"head": "", "cwd": cwd}]
            low_tool = tool.lower(); tool_operation = TOOL_OPERATIONS.get(low_tool)
            if tool_operation and low_tool not in {"bash", "shell"}:
                patches = patch_targets(command); kinds = {k for k, _ in patches}; mutation = None
                if tool_operation == "MUTATE_FILE": mutation = "MIXED" if len(kinds) > 1 else next(iter(kinds)) if kinds else TOOL_MUTATIONS.get(low_tool, "EDIT")
                dedicated_paths = [p for _, p in patches]
                segments = [{"operations": [tool_operation], "mutation": mutation, "paths": dedicated_paths,
                             "mutation_paths": dedicated_paths if mutation else [],
                             "terminal": tool_operation == "NO_REPOSITORY_OPERATION",
                             "read_scope": read_scope("", arguments) if tool_operation == "READ_FILE" else "NOT_APPLICABLE",
                             "external_redirect": False, "cwd": cwd, "rule_ids": ["dedicated_tool"]}]
            operations = [operation for segment in segments for operation in segment["operations"]]
            primary = next((x for x in operations if x != "OTHER_OPERATION"), operations[0] if operations else "OTHER_OPERATION")
            leading = next((segment for segment in segments if primary in segment["operations"]), segments[0] if segments else None)
            terminal = any(s["terminal"] for s in segments); mutations = {s["mutation"] for s in segments if s["mutation"]}
            mutation = "MIXED" if len(mutations) > 1 else next(iter(mutations)) if mutations else "NOT_APPLICABLE"
            viewport = VIEWER_RE.match(command.strip()) is not None; header = FILE_HEADER_RE.search(output)
            shown = header.group(1).strip() if viewport and execution == "executed" and header and NUMBERED_LINE_RE.search(output) else None
            if viewport and execution != "executed": primary, read_mode = "UNKNOWN", "UNKNOWN"
            elif viewport and shown: primary, read_mode = "READ_FILE", "VIEWPORT_CONTINUATION"
            elif viewport: primary, read_mode = "NO_REPOSITORY_OPERATION", "NOT_APPLICABLE"
            elif execution == "parser_unknown": primary, read_mode = "UNKNOWN", "UNKNOWN"
            else: read_mode = "DIRECT_FILE_READ" if primary == "READ_FILE" else "NOT_APPLICABLE"
            raw_paths = argument_paths(arguments) or ([shown] if shown else []) or list((leading or {}).get("paths", []))
            resolved = []
            for candidate in raw_paths:
                path = resolve_path(candidate, (leading or {}).get("cwd", cwd), root)
                if path and path not in resolved: resolved.append(path)
            mutation_paths = []
            for segment in segments:
                if not segment["mutation"]: continue
                for candidate in segment.get("mutation_paths", []) or segment["paths"]:
                    path = resolve_path(candidate, segment.get("cwd", cwd), root)
                    if path and path not in mutation_paths: mutation_paths.append(path)
            if not mutation_paths and mutation != "NOT_APPLICABLE": mutation_paths = list(resolved)
            stateful_edit = any(
                "patch_or_edit" in segment["rule_ids"] and re.match(r"^(?:edit|replace)\b", record["head"])
                for segment, record in zip(segments, records)
            )
            state_candidate = resolved[0] if primary == "READ_FILE" and len(resolved) == 1 else current_file
            stateful_target_used = bool(not mutation_paths and stateful_edit and state_candidate)
            if stateful_target_used: mutation_paths = [str(state_candidate)]
            rule_ids = [rule_id for segment in segments for rule_id in segment["rule_ids"]] + (["stateful_current_file_target"] if stateful_target_used else [])
            if execution == "executed" and primary == "READ_FILE" and len(resolved) == 1:
                current_file = resolved[0]
            elif execution == "executed" and "MUTATE_FILE" in operations and len(mutation_paths) == 1:
                current_file = mutation_paths[0]
            if primary == "UNKNOWN" and viewport:
                target_role, target_role_rule_ids = "UNRESOLVED", ["viewport_unresolved"]
            elif "MUTATE_FILE" in operations:
                target_role, target_role_rule_ids = role_of_with_rules(mutation_paths)
                target_role_rule_ids = ["mutation_destination_precedence", *target_role_rule_ids]
            elif primary == "READ_FILE":
                target_role, target_role_rule_ids = role_of_with_rules(resolved)
            else:
                target_role, target_role_rule_ids = "NOT_APPLICABLE", ["target_role_not_applicable"]
            formal_test_invocation_count = operations.count("RUN_FORMAL_TEST")
            contains_test = formal_test_invocation_count > 0
            if contains_test and execution == "executed":
                test_result, test_result_rule, test_evidence, test_result_conflict = classify_test_result_evidence(output, exit_code)
                if test_result == "unknown" and error is not None:
                    test_result, test_result_rule = "failed", "observation_error"
                    test_evidence = [*test_evidence, "observation_error"]
            else:
                test_result, test_result_rule, test_evidence, test_result_conflict = "NOT_APPLICABLE", "not_applicable", [], False
            explicit_success = True if exit_code == 0 else False if isinstance(exit_code, int) or error is not None else None
            no_observed_failure = None if execution != "executed" else not (error is not None or isinstance(exit_code, int) and exit_code != 0 or test_result == "failed")
            contains_mutation = "MUTATE_FILE" in operations
            if not contains_mutation:
                contains_non_test_repository_path = False
            elif not mutation_paths:
                contains_non_test_repository_path = None
            else:
                contains_non_test_repository_path = any(path_role(path)[0] == "NON_TEST_REPOSITORY_FILE" for path in mutation_paths)
            write_success_evidence = None
            if execution == "executed" and contains_mutation and target_role in {"TEST_FILE", "NON_TEST_REPOSITORY_FILE", "MIXED"}:
                if explicit_success is True:
                    write_success_evidence = "exit_code"
                elif explicit_success is None and no_observed_failure is True:
                    write_success_evidence = "no_observed_failure"
            event_paths = mutation_paths if mutation_paths else resolved
            events.append({
                "event_id": f"{rollout_id}:{step}:{call_index}", "step_index": step, "tool_call_index": call_index,
                "original_tool": tool, "command": command, "execution_status": execution, "pairing_status": pairing,
                "primary_operation": primary, "contained_operations_ordered": operations, "matched_rule_ids": rule_ids,
                "contains_formal_test": contains_test,
                "formal_test_invocation_count": formal_test_invocation_count,
                "contains_multiple_formal_test_invocations": formal_test_invocation_count > 1,
                "contains_mutation": contains_mutation,
                "contains_non_test_repository_path": contains_non_test_repository_path,
                "contains_diff_inspection": "INSPECT_WORKSPACE_CHANGES" in operations,
                "target_role": target_role, "target_role_rule_ids": target_role_rule_ids,
                "mutation_kind": mutation, "read_mode": read_mode,
                "read_scope": (leading or {}).get("read_scope", "NOT_APPLICABLE") if primary == "READ_FILE" else "NOT_APPLICABLE",
                "affected_paths": resolved, "mutation_paths": mutation_paths,
                "external_redirect": any(s["external_redirect"] for s in segments), "exit_code": exit_code,
                "command_succeeded_explicit": explicit_success, "no_observed_failure": no_observed_failure,
                "write_success_evidence": write_success_evidence, "formal_test_result": test_result,
                "formal_test_result_rule": test_result_rule, "formal_test_evidence": test_evidence,
                "formal_test_result_conflict": test_result_conflict,
                "canonical_command_signature": canonical_signature(command, operations, event_paths),
                "semantic_composition_eligible": execution == "executed" and primary in SEMANTIC_OPERATIONS,
                "attempted_action_eligible": not terminal,
            })
    return {"parser_id": PARSER_ID, "codebook_version": CODEBOOK_VERSION,
            "rollout_id": rollout_id, "repository_root": root,
            "repository_root_source": root_source, "events": events, "diagnostics": diagnostics(events)}


def unsuccessful_attempt_state(event: dict[str, Any]) -> bool | None:
    """Return True for observed/rejected failure, False for observed success, else unknown."""
    if event["execution_status"] in {"rejected_invalid_arguments_attempt", "rejected_multi_command_attempt"}:
        return True
    if event["execution_status"] != "executed":
        return None
    if event["no_observed_failure"] is False:
        return True
    if event["no_observed_failure"] is True:
        return False
    return None


def compute_rollout_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute deterministic summaries; writes include known non-test destinations in MIXED calls."""
    attempted = [e for e in events if e["attempted_action_eligible"]]
    semantic = [e for e in events if e["semantic_composition_eligible"]]
    non_test_writes = [
        e for e in events
        if e["execution_status"] == "executed"
        and e["contains_non_test_repository_path"] is True
        and e["no_observed_failure"] is True
    ]
    formal_test_calls = [e for e in events if e["execution_status"] == "executed" and e["contains_formal_test"]]
    formal_test_invocation_count = sum(e["formal_test_invocation_count"] for e in formal_test_calls)
    n = len(semantic); groups = [e["primary_operation"] for e in semantic]
    counts = collections.Counter(groups)
    entropy = (0.0 if len(counts) <= 1 else -sum((value / n) * math.log(value / n) for value in counts.values())) if n else None
    entropy_normalized = None
    if n >= 2 and len(counts) >= 2:
        entropy_normalized = entropy / math.log(len(counts))
    switch_count = sum(a != b for a, b in zip(groups, groups[1:]))
    failed_by_signature = collections.Counter(); signature_repeats = 0
    strict_repeats = 0; previous_signature = None; previous_unsuccessful = False
    for event in attempted:
        signature = event["canonical_command_signature"]
        state = unsuccessful_attempt_state(event)
        if state is False:
            failed_by_signature[signature] = 0
        elif state is True:
            signature_repeats += int(failed_by_signature[signature] > 0)
            failed_by_signature[signature] += 1
        strict_repeats += int(state is True and previous_unsuccessful and signature == previous_signature)
        previous_unsuccessful = state is True
        previous_signature = signature if state is True else None
    productive = [i for i, e in enumerate(attempted) if e in non_test_writes or e["formal_test_result"] == "passed"]
    tail = len(attempted) if not productive else len(attempted) - 1 - max(productive)
    flat = [(i, j, op) for i, e in enumerate(events) for j, op in enumerate(e["contained_operations_ordered"])]
    diffs = [(i, j) for i, j, op in flat if op == "INSPECT_WORKSPACE_CHANGES"]
    write_ids = {id(e) for e in non_test_writes}
    write_positions = [(i, j) for i, e in enumerate(events) for j, op in enumerate(e["contained_operations_ordered"]) if id(e) in write_ids and op == "MUTATE_FILE"]
    last_write = max(write_positions) if write_positions else None
    post_edit_diff = bool(last_write is not None and any(x > last_write for x in diffs))
    passing_tests = [(i, j) for i, e in enumerate(events) for j, op in enumerate(e["contained_operations_ordered"]) if op == "RUN_FORMAL_TEST" and e["formal_test_result"] == "passed"]
    verified_after_write = bool(last_write is not None and any(x > last_write for x in passing_tests))
    unverified_status = "no_write" if last_write is None else "verified" if verified_after_write else "unverified"
    unverified_after_write = unverified_status == "unverified"
    write_count = len(non_test_writes)
    write_share = write_count / n if n else None
    explicit_write_count = sum(e.get("write_success_evidence") == "exit_code" for e in non_test_writes)
    weak_write_count = sum(e.get("write_success_evidence") == "no_observed_failure" for e in non_test_writes)
    return {
        "operational_action_count": len(attempted),
        # Primary-operation exploration and contained test invocations are intentionally non-exclusive.
        "exploration_share": sum(e["primary_operation"] in {"EXPLORE_REPOSITORY", "SEARCH_REPOSITORY", "READ_FILE"} for e in semantic) / n if n else None,
        "read_share": sum(e["primary_operation"] == "READ_FILE" for e in semantic) / n if n else None,
        "search_share": sum(e["primary_operation"] == "SEARCH_REPOSITORY" for e in semantic) / n if n else None,
        "file_mutation_share": sum(e["primary_operation"] == "MUTATE_FILE" for e in semantic) / n if n else None,
        "build_package_share": sum(e["primary_operation"] == "PACKAGE_OR_BUILD" for e in semantic) / n if n else None,
        "other_operation_share": sum(e["primary_operation"] == "OTHER_OPERATION" for e in semantic) / n if n else None,
        "non_test_repository_write_without_observed_failure_count": write_count,
        "non_test_repository_write_without_observed_failure_share": write_share,
        "non_test_repository_write_with_explicit_success_count": explicit_write_count,
        "non_test_repository_write_with_no_observed_failure_only_count": weak_write_count,
        "formal_test_call_count": len(formal_test_calls),
        "formal_test_invocation_count": formal_test_invocation_count,
        "formal_test_call_share": len(formal_test_calls) / n if n else None,
        "formal_test_invocations_per_semantic_action": formal_test_invocation_count / n if n else None,
        "repeated_unsuccessful_command_count": strict_repeats,
        "recurring_unsuccessful_command_count_by_signature": signature_repeats,
        "unproductive_tail_length": tail,
        "action_entropy": entropy, "action_entropy_normalized": entropy_normalized,
        "action_switch_count": switch_count,
        "action_switch_rate": switch_count / (n - 1) if n >= 2 else None,
        "post_edit_diff_inspection": post_edit_diff, "unverified_termination": unverified_status,
        "unverified_termination_after_non_test_repository_write": unverified_after_write,
        "semantic_denominator": n,
    }


def diagnostics(events: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [e for e in events if e["primary_operation"] == "READ_FILE" or e["contains_mutation"]]
    paired = [e for e in events if e["pairing_status"] in {"paired", "ambiguous", "missing"}]
    formal_results = [e for e in events if e["formal_test_result"] != "NOT_APPLICABLE"]
    non_test_writes = [
        e for e in events
        if e["execution_status"] == "executed"
        and e["contains_non_test_repository_path"] is True
        and e["no_observed_failure"] is True
    ]
    other = collections.Counter(
        (e["canonical_command_signature"].split("|")[1].split(" ")[0] if "|" in e["canonical_command_signature"] else "")
        for e in events if e["primary_operation"] == "OTHER_OPERATION"
    )
    count = len(events)
    return {
        "events": count, "applicable_target_events": len(applicable),
        "unknown_primary_rate": sum(e["primary_operation"] == "UNKNOWN" for e in events) / count if count else None,
        "other_operation_rate": sum(e["primary_operation"] == "OTHER_OPERATION" for e in events) / count if count else None,
        "unresolved_target_rate_applicable": sum(e["target_role"] == "UNRESOLVED" for e in applicable) / len(applicable) if applicable else None,
        "mixed_target_role_rate": sum(e["target_role"] == "MIXED" for e in events) / count if count else None,
        "multiple_formal_test_invocation_event_rate": sum(e["formal_test_invocation_count"] > 1 for e in events) / count if count else None,
        "formal_test_result_conflict_rate": sum(e["formal_test_result_conflict"] for e in formal_results) / len(formal_results) if formal_results else None,
        "explicit_write_success_evidence_rate": sum(e["write_success_evidence"] == "exit_code" for e in non_test_writes) / len(non_test_writes) if non_test_writes else None,
        "weak_write_success_evidence_rate": sum(e["write_success_evidence"] == "no_observed_failure" for e in non_test_writes) / len(non_test_writes) if non_test_writes else None,
        "ambiguous_pairing_rate": sum(e["pairing_status"] == "ambiguous" for e in paired) / len(paired) if paired else None,
        "missing_observation_rate": sum(e["pairing_status"] == "missing" for e in paired) / len(paired) if paired else None,
        "compound_event_rate": sum(len(e["contained_operations_ordered"]) > 1 for e in events) / count if count else None,
        "top_unmatched_commands": other.most_common(10),
    }

# Standalone schema adapters
PRIMARY_ADAPTER_ID = "evaluated-final-primary-schema-adapter-v2"
NORMALIZED_EVENTS_ADAPTER_ID = "normalized-events-1.1.0-to-parser-core-v2"
SUPPORTED_PRIMARY_FORMATS = {
    "swelike_tool_calls_v1", "swelike_command_v1",
    "reasoning_state_command_v1", "responses_tool_calls_v1",
}


def _preserve_repository_root(source: dict[str, Any], result: dict[str, Any]) -> None:
    root = source.get("repository_root") or source.get("repo_root")
    if root:
        result["repository_root"] = str(root)
        result["repository_root_source"] = "trajectory"


def normalize_primary_trajectory(trajectory: dict[str, Any], parser_format_id: str) -> dict[str, Any]:
    """Normalize a supported evaluated-final message schema without copying summary fields."""
    if parser_format_id not in SUPPORTED_PRIMARY_FORMATS:
        raise ValueError(f"unsupported parser format: {parser_format_id}")
    result = {"messages": deepcopy(trajectory.get("messages") or [])}
    _preserve_repository_root(trajectory, result)
    for message in result["messages"]:
        if message.get("kind") == "observation" and isinstance(message.get("current_dir"), str):
            message["current_dir"] = message["current_dir"].strip()
        if message.get("kind") != "action":
            continue
        if isinstance(message.get("tool_calls"), list) and message.get("tool_calls"):
            continue
        blocks = COMMAND_BLOCK_RE.findall(str(message.get("content") or ""))
        if len(blocks) > 1:
            command = next((block for block in reversed(blocks) if block.strip()), "[COLLAPSED_MULTIPLE_COMMAND_BLOCKS]")
            message["tool_calls"] = [{
                "id": f"synthetic-command:{message.get('id', '')}",
                "function": {"name": "shell", "arguments": {"command": command, "_multiple_command_blocks": True}},
            }]
    return result


def _normalized_text(value: Any) -> str:
    return _normalized_result(value)[0]


def _normalized_result(value: Any) -> tuple[str, int | None, bool]:
    """Project structured shell results without hiding newlines or nested exit evidence."""
    if value is None:
        return "", None, False
    if isinstance(value, str):
        return value, None, False
    nested_error = isinstance(value, dict) and isinstance(value.get("failure"), dict)
    if isinstance(value, dict) and isinstance(value.get("failure"), dict):
        payload = value["failure"]
    elif isinstance(value, dict) and isinstance(value.get("success"), dict):
        payload = value["success"]
    else:
        payload = value
    exit_code = None
    if isinstance(payload, dict):
        for key in ("exit_code", "exitCode", "returncode", "return_code", "status_code"):
            candidate = payload.get(key)
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                exit_code = candidate
                break
            if isinstance(candidate, str) and re.fullmatch(r"-?\d+", candidate.strip()):
                exit_code = int(candidate)
                break
        if payload.get("interleavedOutput") is not None and payload.get("interleavedOutput") != "":
            values = [str(payload["interleavedOutput"])]
        else:
            values = [str(payload[key]) for key in ("output", "stdout", "stderr", "message", "partial_output") if payload.get(key) is not None and payload.get(key) != ""]
        if values:
            return "\n".join(dict.fromkeys(values)), exit_code, nested_error
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")), exit_code, nested_error


def normalize_event_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Normalize schema 1.1.0 tool-call/tool-result events into parser-core messages."""
    if trajectory.get("schema_version") != "1.1.0" or not isinstance(trajectory.get("events"), list):
        raise ValueError(f"unsupported normalized schema: {trajectory.get('schema_version')!r}")
    raw = trajectory["events"]
    seqs = []
    for event in raw:
        seq = event.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise ValueError(f"invalid event seq: {seq!r}")
        seqs.append(seq)
    if len(seqs) != len(set(seqs)):
        raise ValueError("duplicate normalized event seq")
    size = max(seqs, default=-1) + 1
    if size > 10 * len(raw):
        raise ValueError(f"normalized event seq is too sparse: size={size}, events={len(raw)}")
    messages = [{"kind": "metadata", "id": f"normalized-gap:{i}"} for i in range(size)]
    parents = {}
    for event in raw:
        if event.get("type") != "tool_call":
            continue
        seq = event["seq"]
        call_id = str(event.get("tool_call_id") or f"normalized-missing-call:{seq}")
        action_id = f"normalized-action:{seq}"
        parents[call_id] = action_id
        messages[seq] = {
            "kind": "action", "id": action_id,
            "tool_calls": [{"id": call_id, "function": {"name": str(event.get("tool") or "unknown"), "arguments": deepcopy(event.get("input"))}}],
        }
    for event in raw:
        if event.get("type") != "tool_result":
            continue
        seq = event["seq"]
        call_id = str(event.get("tool_call_id") or "")
        status = str(event.get("status") or "").lower()
        content, nested_exit_code, nested_error = _normalized_result(event.get("content"))
        message = {
            "kind": "observation", "id": f"normalized-observation:{seq}",
            "tool_call_id": call_id, "parent_id": parents.get(call_id, ""), "output": content,
        }
        event_exit_code = event.get("exit_code") if event.get("exit_code") is not None else nested_exit_code
        if event_exit_code is not None:
            message["exit_code"] = event_exit_code
        cwd = event.get("current_dir", event.get("cwd"))
        if isinstance(cwd, str) and cwd.strip():
            message["current_dir"] = cwd.strip()
        if nested_error or event.get("is_error") is True or status in {"error", "failed", "failure", "rejected"}:
            message["error"] = {
                "type": "structured_failure" if nested_error else status or "error", "message": content,
                **({"exit_code": event_exit_code} if event_exit_code is not None else {}),
            }
        messages[seq] = message
    result = {"messages": messages}
    _preserve_repository_root(trajectory, result)
    return result


def normalize_input(
    trajectory: dict[str, Any], schema: str = "auto", parser_format_id: str | None = None,
) -> dict[str, Any]:
    """Apply an explicit adapter, or conservatively detect normalized schema 1.1.0."""
    if schema == "normalized-events" or (schema == "auto" and trajectory.get("schema_version") == "1.1.0" and isinstance(trajectory.get("events"), list)):
        return normalize_event_trajectory(trajectory)
    if schema == "evaluated-final":
        format_id = parser_format_id or trajectory.get("parser_format_id")
        if not format_id:
            raise ValueError("--parser-format is required for evaluated-final input")
        return normalize_primary_trajectory(trajectory, str(format_id))
    if schema not in {"auto", "messages"}:
        raise ValueError(f"unsupported input schema: {schema}")
    if not isinstance(trajectory.get("messages"), list):
        raise ValueError("input does not contain a messages list")
    result = {"messages": deepcopy(trajectory["messages"])}
    _preserve_repository_root(trajectory, result)
    return result


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trajectory", type=Path, help="JSON trajectory file")
    cli.add_argument("--schema", choices=("auto", "messages", "evaluated-final", "normalized-events"), default="auto")
    cli.add_argument("--parser-format", choices=tuple(sorted(SUPPORTED_PRIMARY_FORMATS)))
    cli.add_argument("--repository-root", help="absolute repository root recorded by trusted trajectory or roster metadata")
    cli.add_argument("--rollout-id")
    cli.add_argument("--include-metrics", action="store_true", help="include deterministic rollout summaries")
    cli.add_argument("--allow-large-integers", action="store_true", help="disable Python's integer-string limit for trusted inputs")
    cli.add_argument("--version", action="version", version=PARSER_ID)
    args = cli.parse_args()
    if args.allow_large_integers and hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    normalized = normalize_input(trajectory, args.schema, args.parser_format)
    root = args.repository_root or normalized.get("repository_root") or normalized.get("repo_root")
    root_source = "cli" if args.repository_root else str(normalized.get("repository_root_source") or "missing")
    if not root:
        if args.include_metrics:
            cli.error("--include-metrics requires --repository-root or repository_root in the trajectory")
        print("warning: repository root is missing; applicable file targets may be UNRESOLVED", file=sys.stderr)
    parsed = parse_trajectory(normalized, args.rollout_id or args.trajectory.stem, root, root_source)
    if args.include_metrics:
        parsed["rollout_metrics"] = compute_rollout_metrics(parsed["events"])
    print(json.dumps(parsed, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()

