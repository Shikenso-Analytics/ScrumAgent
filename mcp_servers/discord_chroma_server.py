"""MCP Server for semantic search over Discord messages stored in ChromaDB.

Run with: python mcp_servers/discord_chroma_server.py
Transport: stdio (launched as subprocess by MultiServerMCPClient)
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Optional

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

load_dotenv()

server = Server("discord-chroma-search")


def _get_chroma_db() -> Chroma:
    chroma_path = os.getenv("CHROMA_DB_PATH", "resources/chroma")
    collection_name = os.getenv("CHROMA_DB_DISCORD_CHAT_DATA_NAME", "discord_chat_data")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    client = chromadb.PersistentClient(chroma_path)
    client.get_or_create_collection(collection_name)
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


_db: Optional[Chroma] = None


def get_db() -> Chroma:
    global _db
    if _db is None:
        _db = _get_chroma_db()
    return _db


def discord_semantic_search(query: str, max_results: int = 5) -> str:
    """Search for Discord messages semantically similar to the query."""
    db = get_db()
    results = db.similarity_search(query, k=max_results)
    if not results:
        return "No matching Discord messages found."

    lines = []
    for r in results:
        content = r.page_content.replace("\n", " ")
        ts = datetime.fromtimestamp(r.metadata["timestamp"]).strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"{content} (User: {r.metadata['author_name']}, "
            f"Channel: {r.metadata['channel_name']}, Time: {ts})"
        )
    return "\n".join(lines)


def discord_channel_history(
    channel_name: str,
    after_timestamp: Optional[float] = None,
    before_timestamp: Optional[float] = None,
) -> str:
    """Retrieve Discord messages from a specific channel, optionally filtered by time range."""
    db = get_db()
    where_filter: list[dict] = [
        {"source": "discord_chat"},
        {"channel_name": channel_name},
    ]
    if before_timestamp is not None:
        where_filter.append({"timestamp": {"$lte": before_timestamp}})
    if after_timestamp is not None:
        where_filter.append({"timestamp": {"$gte": after_timestamp}})

    results = db.get(where={"$and": where_filter})
    if not results["documents"]:
        return f"No messages found in channel '{channel_name}' for the given time range."

    lines = []
    for content, meta in zip(results["documents"], results["metadatas"]):
        ts = datetime.fromtimestamp(meta["timestamp"]).strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"{content.replace(chr(10), ' ')} "
            f"(User: {meta['author_name']}, Channel: {meta['channel_name']}, Time: {ts})"
        )
    return "\n".join(lines)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="discord_semantic_search",
            description=(
                "Search for Discord messages semantically similar to the query. "
                "Uses vector similarity search over ChromaDB embeddings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (natural language).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results. Defaults to 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="discord_channel_history",
            description=(
                "Retrieve Discord messages from a specific channel, "
                "optionally filtered by time range using Unix timestamps."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_name": {
                        "type": "string",
                        "description": 'Name of the Discord channel or thread (e.g. "#1234 My Thread").',
                    },
                    "after_timestamp": {
                        "type": "number",
                        "description": "Unix timestamp — only return messages after this time.",
                    },
                    "before_timestamp": {
                        "type": "number",
                        "description": "Unix timestamp — only return messages before this time.",
                    },
                },
                "required": ["channel_name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "discord_semantic_search":
        result = discord_semantic_search(
            query=arguments["query"],
            max_results=arguments.get("max_results", 5),
        )
    elif name == "discord_channel_history":
        result = discord_channel_history(
            channel_name=arguments["channel_name"],
            after_timestamp=arguments.get("after_timestamp"),
            before_timestamp=arguments.get("before_timestamp"),
        )
    else:
        result = f"Unknown tool: {name}"

    return [TextContent(type="text", text=result)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
