import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List

import httpx
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from tqdm import tqdm

load_dotenv()

GITEA_BASE_URL = os.environ.get("GITEA_BASE_URL")
GITEA_API_TOKEN = os.environ.get("GITEA_API_TOKEN")


def get_headers() -> Dict[str, str]:
    """
    Get headers for Gitea API requests
    :return: Headers dictionary
    """
    return {
        "Authorization": f"token {GITEA_API_TOKEN}",
        "Accept": "application/json"
    }


def get_user_repos() -> Dict:
    """
    Get the authenticated user's repositories.
    """
    url = f"{GITEA_BASE_URL}/api/v1/user/repos"
    response = httpx.get(url, headers=get_headers())
    response.raise_for_status()
    return response.json()


def get_branches(owner: str, repo: str) -> List[str]:
    """
    Get all branches for a repository.

    """
    url = f"{GITEA_BASE_URL}/api/v1/repos/{owner}/{repo}/branches"
    response = httpx.get(url, headers=get_headers())
    response.raise_for_status()
    branches = response.json()
    return [b['name'] for b in branches]


def get_commits(owner: str, repo: str, since: datetime, branch: str = None) -> List[Dict]:
    """
    Get commits for a repository since a given datetime, optionally by branch.

    :param owner: Repository owner
    :param repo: Repository name
    :param since: Datetime since which to fetch commits
    :param branch: Branch name (optional)
    :return: List of commit objects
    """
    commits_url = f"{GITEA_BASE_URL}/api/v1/repos/{owner}/{repo}/commits"
    page = 1
    limit = 50
    all_commits = []
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    while True:
        params = {
            "page": page,
            "limit": limit
        }
        if branch:
            params["sha"] = branch
        response = httpx.get(commits_url, headers=get_headers(), params=params)
        response.raise_for_status()
        commits = response.json()

        if not commits:
            break

        filtered_commits = []
        for commit in commits:
            commit_date_str = commit["commit"]["committer"]["date"]
            commit_date = datetime.fromisoformat(commit_date_str.replace('Z', '+00:00'))

            if commit_date >= since:
                filtered_commits.append(commit)
            else:
                # Older commit encountered
                break

        all_commits.extend(filtered_commits)
        if len(filtered_commits) < len(commits):
            # We encountered older commits, stop pagination
            break

        page += 1

    return all_commits


def fetch_weekly_commits_per_repo() -> Dict:
    """
    Fetch commit messages for all repositories for the last week.
    """
    # Calculated "since" = 1 week ago
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    since = one_week_ago.replace(microsecond=0).isoformat() + "Z"

    repos = get_user_repos()
    commits_by_repo = {}

    for repo_data in tqdm(repos, desc="Fetching commits"):
        owner = repo_data['owner']['login']
        repo_name = repo_data['name']

        # Get all branches for this repository
        branches = get_branches(owner, repo_name)

        all_messages_by_branch = {}
        seen_shas = set()  # Deduplicate commits by SHA

        for branch in branches:
            all_messages = []
            # if branch not in ["main", "master"]:
            #     # Skip non-default branches
            #     continue
            commits = get_commits(owner, repo_name, since=datetime.fromisoformat(since.replace('Z', '')), branch=branch)
            for commit in commits:
                sha = commit.get('sha', '')
                msg = commit.get('commit', {}).get('message', '')
                if msg and sha not in seen_shas:
                    seen_shas.add(sha)
                    all_messages.append(msg.strip())
            if all_messages:
                all_messages_by_branch[branch] = all_messages

        if all_messages_by_branch:
            full_repo_name = f"{owner}/{repo_name}"
            commits_by_repo[full_repo_name] = all_messages_by_branch

    return commits_by_repo


def summarize_repo_changes(repo_name: str, commits_by_branch: Dict[str, List[str]]) -> str:
    """
    Summarize recent commit messages for a repository.
    :param repo_name: Repository name
    :param commits_by_branch: Dictionary of commit messages by branch
    :return: Summary text
    """
    messages_main_or_master = commits_by_branch.get("main", []) + commits_by_branch.get("master", [])
    messages_other_branches = [msg for branch, msgs in commits_by_branch.items() if branch not in ["main", "master"] for
                               msg in msgs]
    text_block_main_or_master = "\n".join(messages_main_or_master)
    text_block_other_branches = "\n".join(messages_other_branches)

    # Prompt for a shorter, bullet-point summary
    prompt = PromptTemplate(
        input_variables=["repo_name", "commit_messages_main_or_master", "commit_messages_other_branches"],
        template=(
            "You are a concise documentation assistant. Your task is to summarize recent commit messages for the repository '{repo_name}'. "
            "Here is the information:\n\n"
            "**Deployed to Production (main/master):**\n\n"
            "{commit_messages_main_or_master}\n\n"
            "**Work-in-Progress (other branches):**\n\n"
            "{commit_messages_other_branches}\n\n"
            "Summarize the key updates in the following format:\n\n"
            "- Deployed: [List major features, fixes, or improvements deployed to production. Use 'None' if nothing was deployed.]\n"
            "- Work in Progress: [Highlight ongoing developments or features under work. Use 'None' if there are no updates.]\n\n"
            "Ensure the summary is concise, avoids filler words, and is suitable for sharing on Discord. Use natural language, "
            "but make it clear and easy to understand."
        )
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(repo_name=repo_name, commit_messages_main_or_master=text_block_main_or_master,
                     commit_messages_other_branches=text_block_other_branches)


def summarize_overall_changes(summaries: Dict[str, str]) -> str:
    """
    Summarize changes across multiple repositories.
    :param summaries: Dictionary of repository summaries (repo name -> summary text)
    :return: Overall summary text
    """
    repo_summaries_text = "\n\n".join([f"**{r}**:\n{s}" for r, s in summaries.items()])
    # Prompt for a short overarching summary
    prompt = PromptTemplate(
        input_variables=["repo_summaries"],
        template=(
            "Below are summaries of changes for multiple repositories:\n\n"
            "{repo_summaries}\n\n"
            "Create a brief, high-level overview (in bullet points) of the main themes and improvements across all repositories. "
            "If any changes in one repo may impact others (shared libraries, integrations), highlight them briefly."
            "Make the answer be maximum 1200 characters."
        )
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(repo_summaries=repo_summaries_text)


def summarize_overall_changes_non_technical(summaries: Dict[str, str]) -> str:
    """
    Summarize changes across multiple repositories in a non-technical way.
    :param summaries: Dictionary of repository summaries (repo name -> summary text)
    :return: Overall summary text
    """
    repo_summaries_text = "\n\n".join([f"**{r}**:\n{s}" for r, s in summaries.items()])
    # Prompt for a short overarching summary
    prompt = PromptTemplate(
        input_variables=["repo_summaries"],
        template=(
            "Below are summaries of changes for multiple repositories:\n\n"
            "{repo_summaries}\n\n"
            "Create a brief, overview (2-3 short sentences in simple english) of the main themes and improvements across all repositories "
            "and highlight any potential impacts on particular projects. "
            "Make it very short and non-technical, suitable for sharing with non-technical stakeholders."
        )
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(repo_summaries=repo_summaries_text)
