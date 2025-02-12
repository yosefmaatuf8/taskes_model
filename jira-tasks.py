from jira import JIRA
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Initialize Jira Client
jira = JIRA(
    server=os.getenv("JIRA_SERVER"),
    basic_auth=(os.getenv("JIRA_EMAIL"), os.getenv("ATLASSIAN_API_TOKEN"))
)
PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

# Function to get issue key by summary
def get_issue_key_by_summary(summary):
    """Fetches the issue key for a given issue summary."""
    issues = jira.search_issues(f"project={PROJECT_KEY} AND summary~\"{summary}\"")
    if issues:
        return issues[0].key
    raise ValueError(f"Issue with summary '{summary}' not found.")

# Updated Function to assign an issue to a user
def assign_issue(issue_summary, display_name):
    """Assigns a user to the specified issue using their display name."""
    issue_key = get_issue_key_by_summary(issue_summary)
    try:
        jira.assign_issue(issue_key, display_name)
        print(f"Assigned user '{display_name}' to issue '{issue_summary}' (key: {issue_key}).")
    except Exception as e:
        print(f"Failed to assign issue '{issue_key}' to '{display_name}': {e}")

# Function to create a new issue
def create_issue(summary, description, issue_type='Task'):
    """Creates a new issue in the specified project."""
    issue = jira.create_issue(
        project=PROJECT_KEY,
        summary=summary,
        description=description,
        issuetype={"name": issue_type}
    )
    print(f"Issue '{summary}' created with key '{issue.key}'.")
    return issue

# Function to add a comment to an issue
def comment_on_issue(issue_summary, comment):
    """Adds a comment to the specified issue using the issue summary."""
    issue_key = get_issue_key_by_summary(issue_summary)
    jira.add_comment(issue_key, comment)
    print(f"Added comment to issue '{issue_summary}' (key: {issue_key}): {comment}")

# Function to transition an issue to a new status
def transition_issue(issue_summary, transition_name):
    """Transitions an issue to a new status using the issue summary."""
    issue_key = get_issue_key_by_summary(issue_summary)
    transitions = jira.transitions(issue_key)
    transition_id = next(
        (t["id"] for t in transitions if t["name"].lower() == transition_name.lower()), None
    )
    if not transition_id:
        raise ValueError(f"Transition '{transition_name}' not found for issue '{issue_summary}'.")
    jira.transition_issue(issue_key, transition_id)
    print(f"Issue '{issue_summary}' (key: {issue_key}) transitioned to '{transition_name}'.")

# Function to get all issue details
def get_all_issues_details():
    """Fetches all issue details including users, description, and status."""
    issues = jira.search_issues(f"project={PROJECT_KEY}", maxResults=1000)
    issues_details = {}

    for issue in issues:
        assignee = issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned"
        issues_details[issue.fields.summary] = {
            "key": issue.key,
            "description": issue.fields.description,
            "status": issue.fields.status.name,
            "assignee": assignee,
        }

    return issues_details


if __name__ == "__main__":
    # issue = create_issue("making sure", "just making sure all is well, good and looking fine!!")
    # assign_issue("making sure", "dgreiver")
    # comment_on_issue("connect to jira", "i think i am somewhat closer the the end of this assigment.")
    # transition_issue("making sure", "In Progress")
    print(get_all_issues_details())


