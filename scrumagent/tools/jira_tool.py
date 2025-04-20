import httpx
from base64 import b64encode

# ---- Temel Ayarlar ----
BASE_URL = "http://localhost:8080"
USERNAME = "admin"
PASSWORD = "admin"


def get_auth_header():
    token = b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def get_projects():
    url = f"{BASE_URL}/rest/api/2/project"
    headers = get_auth_header()

    with httpx.Client() as client:
        response = client.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def create_issue(
    project_key: str, summary: str, description: str, issue_type: str = "Bug"
):
    url = f"{BASE_URL}/rest/api/2/issue"
    headers = get_auth_header()
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type},
        }
    }

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=payload)

    if response.status_code == 201:
        return response.json()
    else:
        raise Exception(
            f"Failed to create issue: {response.status_code} - {response.text}"
        )


def get_issue(issue_id_or_key: str):
    url = f"{BASE_URL}/rest/api/2/issue/{issue_id_or_key}"
    headers = get_auth_header()

    with httpx.Client() as client:
        response = client.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to get issue: {response.status_code} - {response.text}"
        )


def add_comment(issue_id_or_key: str, comment: str):
    url = f"{BASE_URL}/rest/api/2/issue/{issue_id_or_key}/comment"
    headers = get_auth_header()
    payload = {"body": comment}

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=payload)

    if response.status_code == 201:
        return response.json()
    else:
        raise Exception(
            f"Failed to add comment: {response.status_code} - {response.text}"
        )


if __name__ == "__main__":
    print("🔑 Authentication Header:")
    print(get_auth_header())
    print("📌 Projects:")
    print(get_projects())

    print("\n🐞 Creating Issue:")
    issue = create_issue(
        "TEST", "Simple Issue via httpx", "Created using simple function-based API."
    )
    print(issue)

    print("\n🔍 Issue Details:")
    print(get_issue(issue["key"]))

    print("\n💬 Adding Comment:")
    print(add_comment(issue["key"], "This is a comment via function-based httpx call."))
