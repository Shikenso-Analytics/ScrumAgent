import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from langgraph.types import interrupt
from pymongo import MongoClient

from scrumagent.agents.agent_state import State
from scrumagent.agents.deepseek_r1_agent import llm_agent
from scrumagent.agents.discord_agent import discord_search_agent
from scrumagent.agents.supervisor_agent import supervisor_node
from scrumagent.agents.taiga_agent import taiga_agent
from scrumagent.agents.web_agent import research_agent
from scrumagent.tools.timeframe_parser_tool import interpret_timeframe_tool, current_timestamp_tool

load_dotenv()

ACTIVATE_DEEPSEEK = os.getenv("ACTIVATE_DEEPSEEK", "").lower() in ("true", "1", "yes", "on")

def human_input_node(state: State) -> Command[Literal["supervisor"]]:
    """Wait for user input and return to the supervisor node."""

    # It doesn't work like expected. It doesn't wait for the user input.
    # But it continues with the next invoke, so its fine I guess.
    human_message = interrupt("human_input")
    return Command(
        update={
            "messages": [{
                "role": "human",
                "content": human_message,
            }]
        },
        goto="supervisor",
    )


# def coder_node(state: State) -> Command[Literal["supervisor"]]:
#     result = code_agent.invoke(state)
#     return Command(
#         update={
#             "messages": [
#                 HumanMessage(content=result["messages"][-1].content, name="coder")
#             ]
#         },
#         goto="supervisor",
#     )

def web_node(state: State) -> Command[Literal["supervisor"]]:
    """Invoke the research agent and pass the result back to the supervisor."""

    result = research_agent.invoke(state)
    print(f"Web Agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="web_browser")
            ]
        },
        goto="supervisor",
    )


def llm_node(state: State) -> Command[Literal["supervisor"]]:
    """Invoke the DeepSeek LLM agent and return its response."""

    result = llm_agent.invoke(state)
    print(f"Deepseek response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="deepseek")
            ]
        },
        goto="supervisor",
    )


def discord_search_node(state: State) -> Command[Literal["supervisor"]]:
    """Invoke the Discord search agent and return its response."""

    result = discord_search_agent.invoke(state)
    print(f"Discord Agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="discord")
            ]
        },
        goto="supervisor",
    )


def taiga_node(state: State) -> Command[Literal["supervisor"]]:
    """Invoke the Taiga agent and forward its response."""

    print("Taiga Agent invoked state: " + state["messages"][-1].content)
    result = taiga_agent.invoke(state)
    print(f"Taiga Agent response: {result['messages'][-1].content}")
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="taiga")
            ]
        },
        goto="supervisor",
    )


def time_parser_node(state: State) -> Command[Literal["supervisor"]]:
    """Parse natural language timeframes in the user's last message."""
    # 1) Extract the user query (raw_timeframe) from 'state'
    #    For example, if the last message from the user was "parse '2 weeks ago'".
    user_message = state["messages"][-1].content

    # 2) Call the timeframe parser
    timestamp_str = interpret_timeframe_tool(user_message)

    # 3) Return to supervisor with the result
    return Command(
        goto="supervisor",
        update={
            "messages": [
                AIMessage(content=timestamp_str, name="time_parser")
            ]
        }
    )


def current_timestamp_node(state: State) -> Command[Literal["supervisor"]]:
    """Return the current timestamp via the helper tool."""

    # 1) Call the current timestamp tool
    timestamp_str = current_timestamp_tool()

    # 2) Return to supervisor with the result
    return Command(
        goto="supervisor",
        update={
            "messages": [
                AIMessage(content=timestamp_str, name="time_parser")
            ]
        }
    )


def build_graph():
    """Compile and return the multi‑agent LangGraph."""
    # https://langchain-ai.github.io/langgraph/concepts/persistence/#using-in-langgraph
    # TODO!: Add the in-memory store to the graph + search for the in-memory store.

    builder = StateGraph(MessagesState)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("web_browser", web_node)
    builder.add_node("discord", discord_search_node)
    # builder.add_node("coder", coder_node)
    if ACTIVATE_DEEPSEEK:
        builder.add_node("deepseek", llm_node)
    builder.add_node("human_input", human_input_node)
    builder.add_node("taiga", taiga_node)
    # builder.add_node("time_parser", time_parser_node)

    # checkpointer = MemorySaver()
    # checkpointer = PostgresSaver(ConnectionPool())

    MONGO_DB_URL = os.getenv("MONGO_DB_URL")
    if MONGO_DB_URL:
        # https://langchain-ai.github.io/langgraph/how-tos/persistence_mongodb/
        ## https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/#use-sync-connection

        client = MongoClient(MONGO_DB_URL)
        checkpointer = MongoDBSaver(client)
        # checkpointer.setup()
    else:
        checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)  # , store=in_memory_store)

    return graph
