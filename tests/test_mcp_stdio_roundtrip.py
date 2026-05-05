import json
import sys

import pytest

EXPECTED_TOOLS = {
    "cartograph.mcp_servers.corpus_reader_bitext": {
        "list_intents",
        "fetch_examples",
        "get_metadata",
        "get_mode",
    },
    "cartograph.mcp_servers.corpus_reader_stackexchange": {
        "fetch_questions",
        "get_mode",
    },
    "cartograph.mcp_servers.corpus_reader_github": {
        "fetch_issues",
        "get_mode",
    },
    "cartograph.mcp_servers.eval_io_adk": {
        "read_evalset",
        "write_evalset",
    },
    "cartograph.mcp_servers.eval_io_git": {
        "read_yaml",
        "read_jsonl",
        "write_yaml",
        "write_jsonl",
    },
    "cartograph.mcp_servers.coverage_state": {
        "list_audits",
        "get_coverage",
        "get_uncovered_regions",
        "get_decision_log",
        "get_generated_cases",
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(("module", "expected"), EXPECTED_TOOLS.items())
async def test_mcp_stdio_servers_expose_expected_tools(module, expected):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", module])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert expected <= {tool.name for tool in tools.tools}


@pytest.mark.asyncio
async def test_eval_io_git_stdio_round_trip(tmp_path):
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    path = tmp_path / "cases.jsonl"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "cartograph.mcp_servers.eval_io_git"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {"read_jsonl", "write_jsonl"} <= {tool.name for tool in tools.tools}

            write_result = await session.call_tool(
                "write_jsonl",
                {"path": str(path), "cases": [{"id": "a"}]},
            )
            assert not write_result.isError

            read_result = await session.call_tool("read_jsonl", {"path": str(path)})
            assert not read_result.isError
            assert read_result.structuredContent == {"result": [{"id": "a"}]}
            assert json.loads(read_result.content[0].text) == {"id": "a"}
