"""Tests for the ScrumAgent class."""

import os
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

os.environ.setdefault("CHROMA_DB_PATH", "resources/chroma")
os.environ.setdefault("CHROMA_DB_DISCORD_CHAT_DATA_NAME", "discord-test")
os.environ.setdefault("OPENAI_API_KEY", "test-key")


class TestScrumAgentInit(unittest.TestCase):
    """Test ScrumAgent initialization without MCP connections."""

    def test_agent_import(self):
        """ScrumAgent class can be imported."""
        from scrumagent.agent import ScrumAgent
        agent = ScrumAgent()
        self.assertIsNone(agent._graph)

    def test_graph_raises_before_start(self):
        """Accessing graph before start() raises RuntimeError."""
        from scrumagent.agent import ScrumAgent
        agent = ScrumAgent()
        with self.assertRaises(RuntimeError):
            _ = agent.graph

    def test_load_mcp_config(self):
        """MCP config YAML loads correctly."""
        from scrumagent.agent import _load_mcp_config
        config = _load_mcp_config()
        self.assertIn("taiga", config)
        self.assertIn("github", config)
        self.assertIn("discord_chroma", config)
        self.assertIn("discord_api", config)
        for name, cfg in config.items():
            self.assertIn("transport", cfg)
            self.assertEqual(cfg["transport"], "stdio")

    def test_build_web_tools(self):
        """Web tools are created (at least DDG should always work)."""
        from scrumagent.agent import _build_web_tools
        tools = _build_web_tools()
        # At minimum DuckDuckGo should be available; others depend on optional deps
        self.assertGreaterEqual(len(tools), 1)
        tool_types = [type(t).__name__ for t in tools]
        self.assertIn("DuckDuckGoSearchResults", tool_types)

    def test_system_prompt_content(self):
        """System prompt contains key sections."""
        from scrumagent.agent import SYSTEM_PROMPT
        self.assertIn("Taiga Management", SYSTEM_PROMPT)
        self.assertIn("Discord Intelligence", SYSTEM_PROMPT)
        self.assertIn("GitHub Insights", SYSTEM_PROMPT)
        self.assertIn("Web Research", SYSTEM_PROMPT)


class TestMCPConfig(unittest.TestCase):
    """Test MCP configuration loading."""

    def test_taiga_config(self):
        from scrumagent.agent import _load_mcp_config
        config = _load_mcp_config()
        taiga = config["taiga"]
        self.assertEqual(taiga["command"], "python")
        self.assertIn("-m", taiga["args"])
        self.assertIn("langchain_taiga.mcp_server", taiga["args"])

    def test_github_config(self):
        from scrumagent.agent import _load_mcp_config
        config = _load_mcp_config()
        github = config["github"]
        self.assertEqual(github["command"], "docker")
        self.assertIn("ghcr.io/github/github-mcp-server", github["args"])

    def test_discord_chroma_config(self):
        from scrumagent.agent import _load_mcp_config
        config = _load_mcp_config()
        dc = config["discord_chroma"]
        self.assertEqual(dc["command"], "python")
        self.assertIn("discord_chroma_server.py", dc["args"][0])


if __name__ == "__main__":
    unittest.main()
