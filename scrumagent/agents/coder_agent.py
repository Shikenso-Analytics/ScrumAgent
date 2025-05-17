from typing import Annotated

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

repl = PythonREPL()


@tool
def python_repl_tool(
        code: Annotated[str, "The python code to execute to generate your chart."],
) -> str:
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    print(f"Executing code: {code}")
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

code_agent = create_react_agent(llm, tools=[python_repl_tool],
                                state_modifier="You are a coder. You may do math and execute code.")
