# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain.messages import AIMessage

from bugbug.tools.patch_summarization.agent import PatchSummarizationTool


def _make_tool(content):
    pytest.importorskip("langchain")

    tool = PatchSummarizationTool.__new__(PatchSummarizationTool)
    tool.target_software = "Mozilla Firefox"
    tool.agent = MagicMock()
    tool.agent.invoke = MagicMock(
        return_value={"messages": [AIMessage(content=content)]}
    )
    return tool


def _make_patch():
    return SimpleNamespace(
        raw_diff="diff",
        patch_set=[],
        has_bug=False,
        patch_title="title",
        patch_description="description",
    )


def test_run_handles_string_content():
    tool = _make_tool("This patch fixes the thing.")
    assert tool.run(_make_patch()) == "This patch fixes the thing."


def test_run_handles_list_content_blocks():
    # Some models return `.content` as a list of content blocks (e.g. when
    # emitting a reasoning block alongside text) rather than a plain string.
    # See https://mozilla.sentry.io/issues/REVIEWHELPER-API-2M/ for the
    # AttributeError this caused when the code assumed `.content` was always
    # a string.
    tool = _make_tool(
        [
            {"type": "reasoning", "reasoning": "thinking about the patch..."},
            {"type": "text", "text": "This patch fixes the thing."},
        ]
    )
    assert tool.run(_make_patch()) == "This patch fixes the thing."


def test_run_strips_budget_token_marker():
    tool = _make_tool("This patch fixes the thing.<budget:token count=123>")
    assert tool.run(_make_patch()) == "This patch fixes the thing."
