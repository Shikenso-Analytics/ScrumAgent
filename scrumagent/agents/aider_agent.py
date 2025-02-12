# Example: scrumagent/agents/aider_agent.py

import logging
from pathlib import Path
from typing import Literal

# Import the classes you need from Aider
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Command

from scrumagent.agents.agent_state import State

mod_path = Path(__file__).parent

load_dotenv()

logger = logging.getLogger(__name__)

fnames = [mod_path / "agent_state.py", mod_path / "supervisor_agent.py", mod_path / "web_agent.py", mod_path / "test.py"]

# Initialize a single coder instance at module-level (optional)
# This keeps state across multiple calls if you like
# (Set dry_run=True to avoid writing files)
io = InputOutput(yes=True)
model = Model("o3-mini")  # or 'gpt-3.5-turbo', whichever is available
coder = Coder.create(main_model=model, io=io, dry_run=True, )


def aider_node(state: State) -> Command[Literal["supervisor"]]:
    """
    A node that uses Aider to respond to coding-related queries.
    The user message is the last message in state["messages"].
    """
    # 1) Extract the user’s instruction or request
    user_message = state["messages"][-1].content
    logger.info(f"Aider Node received message: {user_message}")

    # 2) Send it to Aider
    response_text = coder.run(user_message)

    # 3) Return a Command to update the conversation state
    #    and go back to the supervisor when done
    return Command(
        goto="supervisor",
        update={
            "messages": [
                AIMessage(content=response_text, name="aider")
            ]
        }
    )


if __name__ == '__main__':
    # Test the agent
    state = State(messages=[AIMessage(content="Refactor the supervisor agent")])
    command = aider_node(state)
    print(command.update["messages"][0].content)
