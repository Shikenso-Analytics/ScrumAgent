import time
from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.types import Command
from typing_extensions import TypedDict

from .agent_state import State

members = ["discord", "human_input", "taiga", "web_browser", "deepseek", "aider"
           # "time_parser",
           ]
options = members + ["FINISH"]

# "  - 'discord_send_message_tool': to send a message to a specific channel or thread.\n\n"
# "When the request involves sending a message (e.g. 'send a chat message …'), "
# "first use 'discord_list_channels_with_threads_tool' to find the channel/thread matching the given name (semantic search), "
# "and then call 'discord_send_message_tool' with the appropriate channel ID and message content."

members_infos = {
    "taiga": (
        "The 'taiga' worker can retrieve and manage issues, statuses, sprints, and user stories from Taiga. "
        "It can also provide detailed info about a specific user story (including tasks or history) "
        "and can update user story fields (description, assigned user, status, watchers) "
        "and create new user stories. "
        "It already has valid credentials and does not require any additional API details from the user."
        "But you allways need to provide the coresponding user story and taiga_slug first.\n\n"
        "If the user wants to list, retrieve, or manage issues, if the user asks for open sprints, "
        "user stories in open sprints, user story details by ID, or needs to update a user story, "
        "route the request to 'taiga'."
    ),
    "discord": (
        "The 'discord' worker is responsible for handling Discord-related requests. "
        "It has access to multiple tools:\n"
        "  - 'discord_search_tool': to search for posts in the Discord server.\n"
        "  - 'discord_channel_msgs_tool' and 'discord_get_recent_messages_tool': to retrieve older or recent messages.\n"
        "  - 'discord_list_channels_with_threads_tool': to list all channels along with their active threads. "
        "    This tool can be used to locate a channel or thread by name and find its ID.\n"

    ),
    "human_input": (
        "The 'human_input' worker can be used to ask the user for additional information if the request is unclear or missing details. "
    ),
    "deepseek": (
        "The 'deepseek' agent specializes in advanced reasoning, abstract thinking, and decomposing complex tasks into manageable steps. "
        "It excels at analyzing intricate problems, identifying patterns, and generating strategic solutions across diverse domains. "
        "Designed for depth over speed, it prioritizes logical rigor, contextual understanding, and creative problem-solving, making it ideal for scenarios requiring nuanced analysis, long-term planning, or synthesis of ambiguous information. "
        "It collaborates effectively with other agents by providing structured insights, hypotheses, and step-by-step breakdowns to support system-wide objectives."
    ),
    "web_browser": (
        "The 'web_browser' worker can use the DuckDuckGo search engine to search the web for important information, "
        "ArXiv for research papers, YouTube, and Wikipedia. It may also navigate to webpages."
    ),
    "aider": (
        "The 'aider' worker is a coding assistant that takes user instructions "
        "and provides AI-generated code suggestions or refactoring ideas. "
        "It does not directly modify files."
    ),
    # "time_parser": (
    #     "The 'time_parser' worker can interpret timeframe strings in any language, such as "
    #     "'today', 'yesterday', 'this morning', '2 weeks ago', or 'last Monday', "
    #     "and return a timestamp or date range. Whenever the user references a timeframe, "
    #     "like 'today' or '2 weeks ago', route the request to 'time_parser'."
    # ),
}

members_specs = '\n\n'.join(members_infos.values())

system_prompt = f"""
You are a supervisor tasked with orchestrating a conversation among the following workers: {members}.
Your job is to:
  1. Read and understand the user's request.
  2. Determine which worker(s) can best fulfill the request or sub-parts of it.
  3. Send sub-requests to each relevant worker and gather their partial responses.
  4. Synthesize a complete, coherent final answer for the user by combining all relevant partial results.
  5. Provide this final answer clearly, then respond with "FINISH" at the very end and stop.

Each worker is capable of specific tasks, as described below:

{members_specs}

Instructions:
  • If the request is unclear or missing critical info, first route the conversation to 'human_input' to clarify.
  • If the user references tasks in Taiga (issues, sprints, statuses, user stories), use 'taiga'.
  • If the user wants to search or retrieve messages from Discord, use 'discord'.
  • If the user needs a web or research query, use 'web_browser'.
  • If the user needs help with coding or code suggestions, use 'aider'.
  • If you can answer the user's request directly (e.g., a general question like "tell me a joke"),
    then do so yourself by setting "next": "FINISH" and placing your final answer in "messages".
  • Always combine the results from any involved workers before giving your final response.
  • Once you have formed the final answer, output it clearly and do not respond further.
  • When you are ready to finalize, you may produce an internal END signal, but do not show the word END in the user-facing answer.

When you respond, produce valid JSON **only**, with two keys:
1. "next" — one of {options}
2. "messages" — a string message.

Your entire output must look like this example (with your own contents):
{{
  "next": "taiga",
  "messages": "Your message here"
}}

If no worker is needed, simply use:
{{
  "next": "FINISH",
  "messages": "Your final answer"
}}

The current time is {datetime.utcnow().isoformat()}.
Unix timestamp: {time.time()}.

Act as a careful orchestrator to ensure each worker is called appropriately, gather all partial results, then formulate a single final response that directly answers the user's request.
"""


# • If the user wants to send or forward a regular message in Discord, use 'discord_llm'.


class Router(TypedDict):
    next: Literal[*options]
    messages: str


# llm = ChatOpenAI(model_name="o3-mini")
llm = ChatOpenAI(model_name="gpt-4o")


# llm = ChatOpenAI(model_name="o1")

def supervisor_node(state: State) -> Command[Literal[*members, END]]:
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = llm.with_structured_output(Router).invoke(messages)
    print(f"Supervisor response: {response}")

    goto = response["next"]

    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto, "messages": [
        AIMessage(content=response["messages"], name="supervisor")
    ]})
