"""Tests for the Discord Chroma MCP server."""

import os
import unittest

os.environ.setdefault("CHROMA_DB_PATH", "resources/chroma")
os.environ.setdefault("CHROMA_DB_DISCORD_CHAT_DATA_NAME", "discord-test")
os.environ.setdefault("OPENAI_API_KEY", "test-key")


class TestDiscordChromaMCPServer(unittest.TestCase):
    """Test the Discord Chroma MCP server module."""

    def test_server_import(self):
        """MCP server module can be imported."""
        from mcp_servers.discord_chroma_server import server
        self.assertEqual(server.name, "discord-chroma-search")

    def test_tools_registered(self):
        """Both tool functions are callable."""
        from mcp_servers.discord_chroma_server import (
            discord_semantic_search,
            discord_channel_history,
        )
        self.assertTrue(callable(discord_semantic_search))
        self.assertTrue(callable(discord_channel_history))


if __name__ == "__main__":
    unittest.main()
