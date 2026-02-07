"""Single ScrumAgent powered by MCP tools via LangGraph ReAct pattern."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import sentry_sdk
from langchain_core.runnables import RunnableConfig

import yaml
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

try:
    from langchain_taiga import get_wiki_page_tool, list_wiki_pages_tool
except ImportError:
    get_wiki_page_tool = None
    list_wiki_pages_tool = None

load_dotenv()

mod_path = Path(__file__).parent

# ── System prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an AI Scrum Master assistant. You help development teams by:

1. **Taiga Management** — Retrieve, create, and update user stories, tasks, issues, \
and comments in Taiga. You also have wiki tools: list pages, read a page, create a new \
page, and update an existing page. Wiki pages use slugs (not ref numbers). \
Always call get_wiki_page_tool before updating to get the current version number \
(required for optimistic locking).

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
    model = os.getenv("SCRUM_AGENT_MODEL", "gpt-5.2")
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


_SKILL_PREFIX = "skills-"


def _slug_to_title(slug: str) -> str:
    """Convert a skill slug to a human-readable title.

    ``skills-wiki-formatting`` → ``Wiki Formatting``
    """
    return slug.removeprefix(_SKILL_PREFIX).replace("-", " ").title()


def _load_skills(project_slug: str) -> str:
    """Load skill definitions from Taiga wiki pages.

    Uses ``list_wiki_pages_tool`` to discover all pages with the
    ``skills-`` prefix, then fetches each one and returns combined content.
    """
    logger = logging.getLogger(__name__)

    if list_wiki_pages_tool is None or get_wiki_page_tool is None:
        logger.warning("langchain-taiga not installed — skipping skills")
        return ""

    try:
        raw = list_wiki_pages_tool.invoke({"project_slug": project_slug})
        pages = json.loads(raw)
    except Exception:
        logger.warning("Failed to list wiki pages — skipping skills", exc_info=True)
        return ""

    if isinstance(pages, dict) and "error" in pages:
        logger.warning("Could not list wiki pages: %s", pages["error"])
        return ""

    skill_slugs = sorted(
        p["slug"] for p in pages
        if p.get("slug", "").startswith(_SKILL_PREFIX)
    )

    if not skill_slugs:
        logger.info("No skill pages found (prefix '%s')", _SKILL_PREFIX)
        return ""

    parts: list[str] = []
    for slug in skill_slugs:
        try:
            raw = get_wiki_page_tool.invoke({
                "project_slug": project_slug,
                "wiki_slug": slug,
            })
            skill = json.loads(raw)
        except Exception:
            logger.warning("Failed to fetch skill '%s' — skipping", slug)
            continue
        if "error" in skill:
            logger.warning("Skill page '%s' not found", slug)
            continue
        skill_content = skill.get("content", "")
        if skill_content:
            title = _slug_to_title(slug)
            parts.append(f"### {title}\n{skill_content}")
            logger.info("Loaded skill: %s", title)

    if not parts:
        return ""

    return (
        "\n\n## Loaded Skills\n"
        "Follow these skill instructions when they apply to your current task.\n\n"
        + "\n\n".join(parts)
    )


class ScrumAgent:
    """Manages the MCP client lifecycle and provides the compiled agent graph."""

    def __init__(self) -> None:
        self._graph = None
        self._llm = None
        self._all_tools: list = []
        self._checkpointer = None
        self._skills_text: str = ""
        self._skills_loaded_at: float = 0
        self._skills_ttl: float = float(os.getenv("SKILLS_TTL_SECONDS", "300"))

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

        self._llm = _build_llm()
        self._checkpointer = _build_checkpointer()
        self._all_tools = mcp_tools + web_tools

        # Initial skills load + graph build
        self._refresh_skills()
        self._build_graph()

    def _refresh_skills(self) -> None:
        """Reload skills from Taiga wiki if TTL has expired."""
        logger = logging.getLogger(__name__)
        now = time.monotonic()
        if now - self._skills_loaded_at < self._skills_ttl:
            return  # cache still valid

        project_slug = os.getenv("TAIGA_PROJECT_SLUG")
        if not project_slug:
            self._skills_text = ""
            self._skills_loaded_at = now
            return

        new_skills = _load_skills(project_slug)
        if new_skills != self._skills_text:
            self._skills_text = new_skills
            if new_skills:
                logger.info("Skills refreshed from project '%s'", project_slug)
            # Rebuild only if graph already exists (not during initial start)
            if self._graph is not None:
                self._build_graph()
        self._skills_loaded_at = now

    def _build_graph(self) -> None:
        """(Re)build the agent graph with current skills."""
        system_prompt = SYSTEM_PROMPT + self._skills_text
        self._graph = create_agent(
            self._llm,
            self._all_tools,
            system_prompt=system_prompt,
            checkpointer=self._checkpointer,
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

    def invoke(self, messages: list, config: RunnableConfig) -> dict:
        """Synchronous invoke — refreshes skills if TTL expired."""
        self._refresh_skills()
        with sentry_sdk.start_transaction(op="ai.agent", name="ScrumAgent.invoke"):
            return self.graph.invoke({"messages": messages}, config)

    async def ainvoke(self, messages: list, config: RunnableConfig) -> dict:
        """Async invoke — refreshes skills if TTL expired."""
        self._refresh_skills()
        with sentry_sdk.start_transaction(op="ai.agent", name="ScrumAgent.ainvoke"):
            return await self.graph.ainvoke({"messages": messages}, config)
