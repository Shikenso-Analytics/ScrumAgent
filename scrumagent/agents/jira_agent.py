import os

from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper

from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/integrations/tools/jira.ipynb#scrollTo=34bb5968


jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)


llm = ChatOpenAI(model_name="gpt-4o")

# tools ['jql_query', 'get_projects', 'create_issue', 'catch_all_jira_api', 'create_confluence_page']
jira_agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# jira_agent.run("make a new issue in project test to remind me to make more fried rice")
