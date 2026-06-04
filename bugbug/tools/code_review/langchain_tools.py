# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

"""LangGraph tools for code review agent."""

from dataclasses import dataclass, field
from functools import cache
from logging import getLogger
from typing import Optional

import httpx
import tenacity
from langchain.tools import tool
from langgraph.runtime import get_runtime
from searchfox import AsyncSearchfoxClient

from bugbug.tools.code_review.data_types import Skill, SkillLoadError
from bugbug.tools.core.platforms.base import Patch
from bugbug.tools.core.platforms.patch_apply import get_file_after_stack

logger = getLogger(__name__)

_retry = tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)


def _tool_error(message: str, *, fatal: bool = False) -> str:
    prefix = "Fatal" if fatal else "Warning"
    return f"{prefix}: {message}"


@dataclass
class CodeReviewContext:
    patch: Patch


@cache
def _get_client() -> AsyncSearchfoxClient:
    return AsyncSearchfoxClient()


async def _fetch_file(
    path: str,
    revision: Optional[str],
    client: AsyncSearchfoxClient,
    patch: Patch,
) -> str:
    try:
        return await patch.get_old_file(path)
    except (FileNotFoundError, httpx.HTTPStatusError):
        pass
    if revision:
        try:
            return await _retry(client.get_file_at_revision)(path, revision)
        except Exception:  # searchfox raises plain Exception
            pass
    return await _retry(client.get_file)(path)


@tool
async def expand_context(
    file_path: str,
    start_line: int = 0,
    end_line: int = 0,
) -> str:
    """Retrieve the content of a file, optionally restricted to a line range.

    Omit start_line and end_line to get the full file. When specifying a range,
    be careful to not fill your context window with too much data — request the
    minimum necessary, but do not split a continuous range into multiple requests.

    Args:
        file_path: Repository-relative path, e.g. 'dom/media/webaudio/AudioNode.h'.
        start_line: Starting line number (1-based). Omit (or 0) to start from the beginning.
        end_line: Ending line number (inclusive). Omit (or 0) to read to the end of the file.

    Returns:
        The file content, with line numbers prefixed.
    """
    runtime = get_runtime(CodeReviewContext)
    patch = runtime.context.patch

    warning = None
    try:
        patch_stack = patch.patch_stack
    except ValueError as e:
        warning = f"Could not retrieve the full patch stack ({e}). File content reflects only this patch; please flag this in your review."
        patch_stack = [patch.patch_set]

    revision = await patch.get_base_revision()
    client = _get_client()

    async def fetch(path: str) -> str:
        return await _fetch_file(path, revision, client, patch)

    try:
        file_content = await get_file_after_stack(patch_stack, file_path, fetch)
    except FileNotFoundError:
        return f"Warning: {file_path} was removed by the patch stack."
    except Exception as e:
        return f"Warning: could not retrieve {file_path}: {e}."

    lines = file_content.splitlines()
    start = max(1, start_line) - 1 if start_line else 0
    end = min(len(lines), end_line) if end_line else len(lines)

    line_number_width = len(str(end))
    content = "\n".join(
        f"{i + 1:>{line_number_width}}| {lines[i]}" for i in range(start, end)
    )
    if warning:
        return f"Warning: {warning}\n\n{content}"
    return content


def create_load_skill_tool(skills: list[Skill]):
    skills_by_name = {skill.name: skill for skill in skills}
    available_names = ", ".join(skills_by_name)

    catalog_lines = "\n".join(
        f"- **{skill.name}**: {skill.description}" for skill in skills
    )
    description = (
        "Load the contents of a named skill to guide the review. Use this when "
        "the patch touches an area covered by one of the skills below; otherwise, "
        "do not call it.\n\n"
        "Available skills:\n"
        f"{catalog_lines}\n\n"
        "Args:\n"
        "    name: The name of the skill to load (must match one of the names above).\n\n"
        "Returns:\n"
        "    The skill content as Markdown."
    )

    @tool(description=description)
    async def load_skill(name: str) -> str:
        skill = skills_by_name.get(name)
        if skill is None:
            return f"Unknown skill '{name}'. Available: {available_names}."

        try:
            return await skill.load()
        except SkillLoadError:
            logger.exception("Failed to load skill '%s'", name)
            return f"Failed to load skill '{name}'. Please proceed without it."

    return load_skill


def _parse_line_range(spec: str, total: int) -> tuple[int, int]:
    """Parse '10-20', '10-', '-20', '10' into a (start, end) index pair."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        start = (int(lo) - 1) if lo else 0
        end = int(hi) if hi else total
    else:
        n = int(spec)
        start, end = n - 1, n
    return max(0, start), min(total, end)


def _parse_blame_lines(spec: str) -> list[int]:
    """Parse '10,20,30' or '10-20' into a list of 1-based line numbers."""
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in spec.split(",")]


SEARCHFOX_OPERATIONS = {
    "query": "query",
    "id": "id",
    "define": "define",
    "calls-from": "calls_from",
    "calls-to": "calls_to",
    "calls-between": "calls_between",
    "can-gc": "can_gc",
    "function-at": "function_at",
    "get-file": "get_file",
    "field-layout": "field_layout",
    "blame": "blame",
}

SEARCHFOX_PRIMARY_FIELDS = tuple(SEARCHFOX_OPERATIONS.values())

SEARCHFOX_LANG_FLAGS = {
    "cpp",
    "c",
    "js",
    "webidl",
    "java",
    "kotlin",
    "rust",
    "python",
    "html",
    "css",
}


@dataclass
class SearchfoxCommand:
    query: str | None = None
    id: str | None = None
    define: str | None = None
    calls_from: str | None = None
    calls_to: str | None = None
    calls_between: str | None = None
    can_gc: str | None = None
    function_at: str | None = None
    get_file: str | None = None
    field_layout: str | None = None
    blame: str | None = None
    path: str | None = None
    regexp: bool = False
    case: bool = False
    limit: int = 50
    context: int | None = None
    lines: str | None = None
    depth: int = 2
    langs: list[str] = field(default_factory=list)
    tests: str | None = None

    @property
    def operation(self) -> str | None:
        operations = self.operations
        if len(operations) == 1:
            return operations[0]
        return None

    @property
    def operations(self) -> list[str]:
        return [
            field
            for field in SEARCHFOX_PRIMARY_FIELDS
            if getattr(self, field) is not None
        ]


def _parse_searchfox_command(command: str) -> SearchfoxCommand:
    import shlex

    args = shlex.split(command)
    parsed = SearchfoxCommand()
    i = 0

    while i < len(args):
        token = args[i]
        if token in SEARCHFOX_OPERATIONS and i + 1 < len(args):
            setattr(parsed, SEARCHFOX_OPERATIONS[token], args[i + 1])
            i += 2
        elif token in {"path", "lines"} and i + 1 < len(args):
            setattr(parsed, token, args[i + 1])
            i += 2
        elif token in {"depth", "limit", "context"} and i + 1 < len(args):
            setattr(parsed, token, int(args[i + 1]))
            i += 2
        elif token == "regexp":
            parsed.regexp = True
            i += 1
        elif token == "case-sensitive":
            parsed.case = True
            i += 1
        elif token == "exclude-tests":
            parsed.tests = "exclude"
            i += 1
        elif token == "only-tests":
            parsed.tests = "only"
            i += 1
        elif token in SEARCHFOX_LANG_FLAGS:
            parsed.langs.append(token)
            i += 1
        else:
            i += 1

    return parsed


def _format_search_results(results) -> str:
    if not results:
        return "No results found."
    return "\n".join(f"{path}:{line}: {content}" for path, line, content in results)


@tool
async def searchfox(command: str) -> str:
    """Search and navigate the Firefox codebase using Searchfox.

    Syntax: <operation> <value> [modifiers...]

    Operations (set exactly one):
      query <text>                   text or regex search
      id <identifier>                exact identifier search
      define <symbol>                full definition of a symbol or class
      calls-from <symbol>            outgoing calls from symbol
      calls-to <symbol>              incoming callers of symbol
      calls-between <symbol>,<symbol> call paths between two symbols
      can-gc <symbol>                check if C++ function can trigger GC
      function-at <file>:<line>      function/class enclosing a line
      get-file <file>                file content
      field-layout <class>           C++ class memory layout
      blame <file>                   commit info for lines; requires lines modifier

    Modifiers:
      path <prefix>      filter results by path prefix
      depth <N>          call graph traversal depth (default 2)
      limit <N>          max results (default 50)
      context <N>        surrounding lines per match
      lines <range>      line range: 10-20, 10, 10-, -20, or 10,20,30 (blame)
      regexp             treat query as regular expression
      case-sensitive     enable case-sensitive matching
      exclude-tests      omit test files
      only-tests         restrict to test files
      cpp  c  js  webidl  java  kotlin  rust  python  html  css   (language filters)

    Examples:
      searchfox("query AudioStream cpp")
      searchfox("define mozilla::dom::AudioContext")
      searchfox("calls-from AudioNode::Connect depth 2")
      searchfox("get-file dom/media/AudioStream.h lines 10-50")
      searchfox("blame dom/media/AudioStream.cpp lines 42,43,44")
      searchfox("query AudioStream path dom/media regexp")
    """
    parsed = _parse_searchfox_command(command)
    client = _get_client()

    if not parsed.operations:
        return _tool_error(
            "No operation set. Use one of: query, id, define, calls-from, "
            "calls-to, calls-between, can-gc, function-at, get-file, field-layout, blame.",
            fatal=True,
        )
    if len(parsed.operations) > 1:
        return _tool_error(
            f"Multiple operations set: {', '.join(parsed.operations)}. Set exactly one.",
            fatal=True,
        )
    op = parsed.operation

    async def _run() -> str:
        match op:
            case "query":
                results = await _retry(client.search)(
                    query=parsed.query,
                    path=parsed.path,
                    langs=parsed.langs or None,
                    tests=parsed.tests,
                    regexp=parsed.regexp,
                    case=parsed.case,
                    limit=parsed.limit,
                    context=parsed.context,
                )
                return _format_search_results(results)

            case "id":
                results = await _retry(client.search)(
                    id=parsed.id,
                    path=parsed.path,
                    langs=parsed.langs or None,
                    tests=parsed.tests,
                    limit=parsed.limit,
                )
                return _format_search_results(results)

            case "define":
                return await _retry(client.get_definition)(parsed.define, parsed.path)

            case "calls_from":
                return await _retry(client.search_call_graph)(
                    calls_from=parsed.calls_from, depth=parsed.depth
                )

            case "calls_to":
                return await _retry(client.search_call_graph)(
                    calls_to=parsed.calls_to, depth=parsed.depth
                )

            case "calls_between":
                calls_between = parsed.calls_between
                assert calls_between is not None
                parts = calls_between.split(",", 1)
                if len(parts) != 2:
                    return _tool_error("calls-between requires 'SymbolA,SymbolB'")
                return await _retry(client.search_call_graph)(
                    calls_between=(parts[0].strip(), parts[1].strip()),
                    depth=parsed.depth,
                )

            case "can_gc":
                results = await _retry(client.get_gc_info)(parsed.can_gc)
                if not results:
                    return "No GC information found. GC analysis is only available for C++ functions."
                out = []
                for pretty, _mangled, gc, path in results:
                    entry = f"{pretty}: {'can GC' if gc else 'cannot GC'}"
                    if path:
                        entry += f" (via {path})"
                    out.append(entry)
                return "\n".join(out)

            case "function_at":
                spec = parsed.function_at
                assert spec is not None
                if ":" not in spec:
                    return _tool_error("function-at requires 'path:line' format")
                file_path, line_str = spec.rsplit(":", 1)
                try:
                    line_num = int(line_str)
                except ValueError:
                    return _tool_error(f"invalid line number: {line_str!r}")
                return await _retry(client.get_function_at_line)(file_path, line_num)

            case "get_file":
                content = await _retry(client.get_file)(parsed.get_file)
                if not parsed.lines:
                    return content
                file_lines = content.splitlines()
                start, end = _parse_line_range(parsed.lines, len(file_lines))
                width = len(str(end))
                return "\n".join(
                    f"{i + 1:>{width}}| {file_lines[i]}" for i in range(start, end)
                )

            case "field_layout":
                return await _retry(client.search_field_layout)(parsed.field_layout)

            case "blame":
                if not parsed.lines:
                    return _tool_error(
                        "blame requires lines (e.g. lines 10,20 or lines 10-20)"
                    )
                line_nums = _parse_blame_lines(parsed.lines)
                results = await _retry(client.get_blame_for_lines)(
                    parsed.blame, line_nums
                )
                if not results:
                    return "No blame information found."
                return "\n".join(
                    f"{ln}: {hash_} ({date}) {message}"
                    for ln, hash_, message, date in results
                )

            case _:
                return _tool_error("unreachable", fatal=True)

    try:
        return await _run()
    except Exception as e:
        logger.error("searchfox error: %s", e)
        return _tool_error(f"searchfox failed: {e}")


SEARCHFOX_TOOLS = [
    expand_context,
    searchfox,
]
