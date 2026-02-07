"""Tests for the ScrumAgent class."""

import json
import os
import time
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


class TestLoadSkills(unittest.TestCase):
    """Test the _load_skills function."""

    @patch("scrumagent.agent.get_wiki_page_tool")
    @patch("scrumagent.agent.list_wiki_pages_tool")
    def test_load_skills_success(self, mock_list, mock_get):
        """Skills are loaded and formatted correctly."""
        from scrumagent.agent import _load_skills

        mock_list.invoke.return_value = json.dumps([
            {"slug": "home"},
            {"slug": "skill-wiki-formatting"},
            {"slug": "skill-daily-standup"},
            {"slug": "other-page"},
        ])
        mock_get.invoke.side_effect = [
            json.dumps({"slug": "skill-daily-standup", "content": "Run daily standups at 9am.", "version": 1}),
            json.dumps({"slug": "skill-wiki-formatting", "content": "Use markdown links.", "version": 1}),
        ]

        result = _load_skills("my-project")

        self.assertIn("## Loaded Skills", result)
        self.assertIn("### Wiki Formatting", result)
        self.assertIn("### Daily Standup", result)
        self.assertIn("Use markdown links", result)
        self.assertIn("Run daily standups", result)
        self.assertEqual(mock_list.invoke.call_count, 1)
        self.assertEqual(mock_get.invoke.call_count, 2)

    @patch("scrumagent.agent.get_wiki_page_tool")
    @patch("scrumagent.agent.list_wiki_pages_tool")
    def test_load_skills_no_skill_pages(self, mock_list, mock_get):
        """Returns empty string when no skill- pages exist."""
        from scrumagent.agent import _load_skills

        mock_list.invoke.return_value = json.dumps([
            {"slug": "home"},
            {"slug": "other-page"},
        ])

        result = _load_skills("my-project")
        self.assertEqual(result, "")
        mock_get.invoke.assert_not_called()

    @patch("scrumagent.agent.get_wiki_page_tool")
    @patch("scrumagent.agent.list_wiki_pages_tool")
    def test_load_skills_partial_failure(self, mock_list, mock_get):
        """Loads available skills even when some pages fail."""
        from scrumagent.agent import _load_skills

        mock_list.invoke.return_value = json.dumps([
            {"slug": "skill-good"},
            {"slug": "skill-bad"},
        ])
        mock_get.invoke.side_effect = [
            json.dumps({"slug": "skill-bad", "error": "Not found."}),
            json.dumps({"slug": "skill-good", "content": "This skill works.", "version": 1}),
        ]

        result = _load_skills("my-project")
        self.assertIn("### Good", result)
        self.assertNotIn("Bad", result)

    @patch("scrumagent.agent.list_wiki_pages_tool")
    def test_load_skills_list_exception(self, mock_list):
        """Returns empty string when listing pages fails."""
        from scrumagent.agent import _load_skills

        mock_list.invoke.side_effect = ConnectionError("Taiga is down")

        result = _load_skills("my-project")
        self.assertEqual(result, "")

    @patch("scrumagent.agent.get_wiki_page_tool")
    @patch("scrumagent.agent.list_wiki_pages_tool")
    def test_load_skills_ignores_non_skill_pages(self, mock_list, mock_get):
        """Only pages with skill- prefix are loaded."""
        from scrumagent.agent import _load_skills

        mock_list.invoke.return_value = json.dumps([
            {"slug": "skill-formatting"},
            {"slug": "home"},
            {"slug": "sprint-retro"},
        ])
        mock_get.invoke.return_value = json.dumps({
            "slug": "skill-formatting",
            "content": "Format content.",
            "version": 1,
        })

        result = _load_skills("my-project")
        self.assertIn("### Formatting", result)
        self.assertEqual(mock_get.invoke.call_count, 1)

    @patch("scrumagent.agent.get_wiki_page_tool", None)
    @patch("scrumagent.agent.list_wiki_pages_tool", None)
    def test_load_skills_tool_none(self):
        """Returns empty string when langchain-taiga is not installed."""
        from scrumagent.agent import _load_skills
        result = _load_skills("my-project")
        self.assertEqual(result, "")


class TestSkillsRefresh(unittest.TestCase):
    """Test the ScrumAgent TTL-based skills refresh."""

    @patch("scrumagent.agent._load_skills", return_value="### Test\nContent")
    @patch.dict(os.environ, {"TAIGA_PROJECT_SLUG": "test-project"})
    def test_refresh_skips_within_ttl(self, mock_load):
        """Second call within TTL does not reload."""
        from scrumagent.agent import ScrumAgent
        agent = ScrumAgent()
        agent._llm = MagicMock()
        agent._all_tools = []
        agent._checkpointer = MagicMock()
        agent._skills_ttl = 300

        agent._refresh_skills()
        self.assertEqual(mock_load.call_count, 1)

        # Second call — should be cached
        agent._refresh_skills()
        self.assertEqual(mock_load.call_count, 1)

    @patch("scrumagent.agent._load_skills", return_value="### Test\nContent")
    @patch.dict(os.environ, {"TAIGA_PROJECT_SLUG": "test-project"})
    def test_refresh_reloads_after_ttl(self, mock_load):
        """Reloads after TTL expires."""
        from scrumagent.agent import ScrumAgent
        agent = ScrumAgent()
        agent._llm = MagicMock()
        agent._all_tools = []
        agent._checkpointer = MagicMock()
        agent._skills_ttl = 0  # expire immediately

        agent._refresh_skills()
        self.assertEqual(mock_load.call_count, 1)

        agent._refresh_skills()
        self.assertEqual(mock_load.call_count, 2)

    @patch("scrumagent.agent.create_agent")
    @patch("scrumagent.agent._load_skills", return_value="### Test\nContent")
    @patch.dict(os.environ, {"TAIGA_PROJECT_SLUG": "test-project"})
    def test_refresh_no_rebuild_same_content(self, mock_load, mock_create):
        """No graph rebuild when skills content hasn't changed."""
        from scrumagent.agent import ScrumAgent
        agent = ScrumAgent()
        agent._llm = MagicMock()
        agent._all_tools = []
        agent._checkpointer = MagicMock()
        agent._skills_ttl = 0
        agent._graph = MagicMock()  # pretend graph exists

        agent._refresh_skills()
        rebuild_count_1 = mock_create.call_count

        # Second call — same content, no rebuild
        agent._refresh_skills()
        self.assertEqual(mock_create.call_count, rebuild_count_1)


if __name__ == "__main__":
    unittest.main()
