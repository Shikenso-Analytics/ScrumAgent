"""Single ScrumAgent powered by MCP tools via LangGraph ReAct pattern."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

mod_path = Path(__file__).parent

# ── System prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an AI Scrum Master assistant. You help development teams by:

1. **Taiga Management** — Retrieve, create, and update user stories, tasks, issues, \
and comments in Taiga. You need the project_slug and entity reference for most operations.

2. **Discord Intelligence** — Search Discord messages semantically, retrieve channel \
history by timeframe, list channels/threads, and send messages.

3. **GitHub Insights** — Browse repositories, list branches, view commits, \
and summarize recent development activity.

4. **Web Research** — Search the web (DuckDuckGo), ArXiv papers, YouTube, \
and Wikipedia for relevant information.

## Guidelines
- Always verify entity existence before updating (use get_entity_by_ref_tool first).
- When asked about recent activity, combine Taiga task data with Discord discussions.
- For daily standups, analyze both Taiga progress and Discord conversations from the last few days.
- Use Discord-friendly Markdown formatting in your responses.
- Be concise but thorough. Provide actionable summaries.
- When referencing Taiga entities, include their reference number and URL.
- If information is unclear or missing, ask the user for clarification.
"""


def _resolve_env(env_map: dict[str, str]) -> dict[str, str]:
    """Resolve ${VAR} references from os.environ, skip unset vars."""
    resolved = {}
    for key, val in env_map.items():
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            env_key = val[2:-1]
            env_val = os.environ.get(env_key)
            if env_val is not None:
                resolved[key] = env_val
        else:
            resolved[key] = str(val)
    return resolved


def _load_mcp_config() -> dict[str, dict[str, Any]]:
    """Load MCP server configuration from YAML."""
    config_path = mod_path / "../config/mcp_config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    mcp_config: dict[str, dict[str, Any]] = {}
    for name, cfg in raw.items():
        transport = cfg.get("transport", "stdio")
        if transport == "stdio":
            entry: dict[str, Any] = {
                "transport": "stdio",
                "command": cfg["command"],
                "args": cfg.get("args", []),
            }
            if "env" in cfg:
                entry["env"] = _resolve_env(cfg["env"])
            mcp_config[name] = entry
        elif transport == "streamable_http":
            mcp_config[name] = {
                "transport": "streamable_http",
                "url": cfg["url"],
            }
    return mcp_config


def _build_web_tools() -> list:
    """Create LangChain community web search tools.

    Tools that fail to import (e.g. missing optional deps) are silently skipped.
    """
    tools = []
    try:
        tools.append(DuckDuckGoSearchResults(max_results=4, output_format="list"))
    except ImportError:
        pass
    try:
        tools.append(ArxivQueryRun())
    except ImportError:
        pass
    try:
        tools.append(YouTubeSearchTool())
    except ImportError:
        pass
    try:
        tools.append(WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()))
    except ImportError:
        pass
    return tools


def _build_llm():
    """Create the LLM based on SCRUM_AGENT_MODEL env var.

    Supported prefixes:
      - "claude*"   → ChatAnthropic
      - "ollama/*"  → ChatOllama (strip prefix)
      - default     → ChatOpenAI
    """
    model = os.getenv("SCRUM_AGENT_MODEL", "gpt-4o")
    temp = float(os.getenv("SCRUM_AGENT_TEMPERATURE", "0"))

    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temp)
    elif model.startswith("ollama/"):
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model.removeprefix("ollama/"), temperature=temp)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model_name=model, temperature=temp)


def _build_checkpointer():
    """Create a checkpointer — MongoDB if configured, else MemorySaver."""
    mongo_url = os.getenv("MONGO_DB_URL")
    if mongo_url:
        from langgraph.checkpoint.mongodb import MongoDBSaver
        from pymongo import MongoClient
        client = MongoClient(mongo_url)
        return MongoDBSaver(client)
    return MemorySaver()


class ScrumAgent:
    """Manages the MCP client lifecycle and provides the compiled agent graph."""

    def __init__(self) -> None:
        self._graph = None

    async def start(self) -> None:
        """Initialize MCP connections and build the agent graph.

        Each MCP server is connected individually so that a single
        misconfigured server (e.g. missing token) does not prevent the
        remaining servers from loading.
        """
        logger = logging.getLogger(__name__)
        mcp_config = _load_mcp_config()

        mcp_tools: list = []
        for name, server_cfg in mcp_config.items():
            try:
                client = MultiServerMCPClient({name: server_cfg})
                tools = await client.get_tools()
                mcp_tools.extend(tools)
                logger.info("MCP server '%s' loaded %d tools", name, len(tools))
            except Exception:
                logger.exception("MCP server '%s' failed to connect — skipping", name)

        web_tools = _build_web_tools()
        all_tools = mcp_tools + web_tools

        llm = _build_llm()
        checkpointer = _build_checkpointer()

        self._graph = create_react_agent(
            llm,
            all_tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )

    async def stop(self) -> None:
        """Shut down MCP connections."""
        self._graph = None

    @property
    def graph(self):
        """The compiled LangGraph agent. Available after start()."""
        if self._graph is None:
            raise RuntimeError("ScrumAgent not started. Call await agent.start() first.")
        return self._graph

    def invoke(self, messages: list, config: dict) -> dict:
        """Synchronous invoke for compatibility with existing code."""
        return self.graph.invoke({"messages": messages}, config)

    async def ainvoke(self, messages: list, config: dict) -> dict:
        """Async invoke."""
        return await self.graph.ainvoke({"messages": messages}, config)
