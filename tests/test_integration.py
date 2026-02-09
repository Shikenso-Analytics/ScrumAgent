"""Integration tests for ScrumAgent with real MCP connections.

These tests require credentials in .env and are skipped if not configured.
Run with: uv run python -m pytest tests/test_integration.py -v
"""

import os

import pytest
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


@pytest.mark.skipif(
    not os.getenv("TAIGA_TOKEN") and not os.getenv("TAIGA_USERNAME"),
    reason="No Taiga credentials",
)
@pytest.mark.asyncio
async def test_taiga_mcp_connection():
    """Connect to Taiga MCP server and verify tools are available."""
    from scrumagent.agent import _load_mcp_config

    full_config = _load_mcp_config()
    taiga_config = {"taiga": full_config["taiga"]}

    client = MultiServerMCPClient(taiga_config)
    tools = await client.get_tools()
    tool_names = [t.name for t in tools]
    assert len(tool_names) > 0, "No Taiga tools returned"
    assert any("taiga" in name.lower() or "project" in name.lower() or "entity" in name.lower() for name in tool_names), (
        f"No Taiga tools found. Available: {tool_names}"
    )


@pytest.mark.skipif(
    not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
    reason="No GitHub token",
)
@pytest.mark.asyncio
async def test_github_mcp_connection():
    """Connect to GitHub MCP server and verify tools are available."""
    from scrumagent.agent import _load_mcp_config

    full_config = _load_mcp_config()
    github_config = {"github": full_config["github"]}

    client = MultiServerMCPClient(github_config)
    tools = await client.get_tools()
    tool_names = [t.name for t in tools]
    assert len(tool_names) > 0, "No GitHub tools returned"


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "test-key",
    reason="No real OpenAI key",
)
@pytest.mark.asyncio
async def test_discord_chroma_mcp_connection():
    """Connect to Discord Chroma MCP server and verify tools are available."""
    from scrumagent.agent import _load_mcp_config

    full_config = _load_mcp_config()
    chroma_config = {"discord_chroma": full_config["discord_chroma"]}

    client = MultiServerMCPClient(chroma_config)
    tools = await client.get_tools()
    tool_names = [t.name for t in tools]
    assert "discord_semantic_search" in tool_names, (
        f"discord_semantic_search not found. Available: {tool_names}"
    )
    assert "discord_channel_history" in tool_names, (
        f"discord_channel_history not found. Available: {tool_names}"
    )


def test_web_tools_all_present():
    """Verify all 4 web tools are created when dependencies are installed."""
    from scrumagent.agent import _build_web_tools

    tools = _build_web_tools()
    tool_types = [type(t).__name__ for t in tools]
    assert len(tools) == 4, f"Expected 4 tools, got {len(tools)}: {tool_types}"
    assert "DuckDuckGoSearchResults" in tool_types
    assert "ArxivQueryRun" in tool_types
    assert "YouTubeSearchTool" in tool_types
    assert "WikipediaQueryRun" in tool_types


def test_build_llm_default():
    """Verify _build_llm creates ChatOpenAI by default."""
    from scrumagent.agent import _build_llm

    llm = _build_llm()
    assert type(llm).__name__ == "ChatOpenAI"


def test_env_propagation_in_config():
    """Verify _load_mcp_config resolves env vars for MCP servers."""
    from scrumagent.agent import _load_mcp_config

    config = _load_mcp_config()

    # Taiga should have resolved env vars (only those that are set)
    if "env" in config.get("taiga", {}):
        for key, val in config["taiga"]["env"].items():
            assert not val.startswith("${"), f"Unresolved env var: {key}={val}"

    # Discord Chroma should have env block
    assert "env" in config.get("discord_chroma", {}), "discord_chroma missing env block"

    # Discord API should have env block
    assert "env" in config.get("discord_api", {}), "discord_api missing env block"
