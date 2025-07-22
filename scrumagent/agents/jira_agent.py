import os
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper


from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/integrations/tools/jira.ipynb#scrollTo=34bb5968


# Initialize Jira API wrapper
jira = JiraAPIWrapper()

# Get Jira tools
toolkit = JiraToolkit.from_jira_api_wrapper(jira)
tools = toolkit.get_tools()

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o")

# Create the LangGraph-compatible agent
jira_agent = create_react_agent(
    llm,
    tools=tools,
    state_modifier=(
        "You are a Jira project management assistant. Use the tools provided below "
        "to interact with Jira effectively. Always follow best practices for project management.\n\n"
        "### Available Tools:\n"
        "- jql_query: Query issues using JQL\n"
        "- get_projects: List all projects\n"
        "- create_issue: Create a new issue in a specified project\n"
        "- catch_all_jira_api: Perform raw Jira API operations\n"
        "- create_confluence_page: Create documentation in Confluence\n\n"
        "### Workflow Guidance:\n"
        "- Confirm project name before creating issues\n"
        "- Use comments to notify users\n"
        "- Respect project configuration and issue types\n"
        "- Report back if an operation fails\n"
    ),
)

"""
Usage Example:
response = jira_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "make a new issue in project test to remind me to make more fried rice",
            }
        ]
    }
)
print(response["messages"][-1].content)
"""
