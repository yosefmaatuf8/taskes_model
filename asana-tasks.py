from asana import Client
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Initialize Configuration
client = Client.access_token(os.getenv('ASANA_ACCESS_TOKEN'))
client.headers["Asana-Enable"] = "new_user_task_lists"

# Replace these with actual IDs from your environment
WORKSPACE_ID = os.getenv('ASANA_WORKSPACE_ID')

def get_user_id_by_name(user_name):
    """Fetches the user ID from a user name."""
    users = client.users.find_all({"workspace": WORKSPACE_ID})
    for user in users:
        print(f"Checking user: {user}")  # For debugging
        if "name" in user and user["name"].lower() == user_name.lower():
            return user["gid"]
    raise ValueError(f"User with name '{user_name}' not found in workspace.")

# Function to get project ID by project name
def get_project_id_by_name(project_name):
    """Fetches the project ID from a project name."""
    projects = client.projects.find_all({"workspace": WORKSPACE_ID})
    for project in projects:
        if project['name'] == project_name:
            return project['gid']
    raise ValueError(f"Project '{project_name}' not found in workspace.")

# Function to get task by name
def get_task_by_name(project_id, task_name):
    """Fetches a task object by its name."""
    tasks = client.tasks.find_by_project(project_id)
    for task in tasks:
        if task['name'] == task_name:
            return task
    raise ValueError(f"Task '{task_name}' not found in project with ID '{project_id}'.")

# Function to create a new task
def create_task(project_name, task_name, task_description):
    """Creates a new task in the specified project."""
    project_id = get_project_id_by_name(project_name)
    task = client.tasks.create_task({
        "name": task_name,
        "notes": task_description,
        "projects": [project_id],
        "workspace": WORKSPACE_ID
    })
    print(f"Task '{task_name}' created in project '{project_name}'.")
    return task

# Function to assign a user to a task
def assign_user_to_task(project_name, task_name, user_name):
    """Assigns a user to the specified task using their name."""
    project_id = get_project_id_by_name(project_name)
    task = get_task_by_name(project_id, task_name)
    user_id = get_user_id_by_name(user_name)
    client.tasks.update_task(task['gid'], {"assignee": user_id})
    print(f"Assigned user '{user_name}' to task '{task_name}'.")

# Function to comment on a task
def comment_on_task(project_name, task_name, comment):
    """Adds a comment to the specified task by name."""
    project_id = get_project_id_by_name(project_name)
    task = get_task_by_name(project_id, task_name)
    client.tasks.add_comment(task['gid'], {"text": comment})
    print(f"Added comment on task '{task_name}': {comment}")


def move_task_to_section(task_name, project_name, section_name):
    """Moves a task to a specific section within a project."""
    # Get the project ID
    project_id = get_project_id_by_name(project_name)

    # Get the task ID
    task = get_task_by_name(project_id, task_name)
    task_id = task['gid']

    # Find the section ID
    sections = client.sections.find_by_project(project_id)
    for section in sections:
        if section['name'].lower() == section_name.lower():
            section_id = section['gid']
            break
    else:
        raise ValueError(f"Section '{section_name}' not found in project '{project_name}'.")

    # Move the task to the section
    client.sections.add_task(section_id, {"task": task_id})
    print(f"Moved task '{task_name}' to section '{section_name}' in project '{project_name}'.")


# Function to fetch all task details in a workspace
def get_all_issues_details(project_name):
    """Fetches all issue details including assignee, description, and status."""
    project_id = get_project_id_by_name(project_name)
    tasks = client.tasks.find_by_project(project_id)
    issues_details = {}

    for task in tasks:
        task_data = client.tasks.find_by_id(task["gid"])

        assignee = task_data["assignee"].get("name", "Unassigned") if task_data.get("assignee") else "Unassigned"

        # Find the section (list) the task belongs to
        memberships = task_data.get("memberships", [])
        list_name = "No List Assigned"
        for membership in memberships:
            if "section" in membership:
                list_name = membership["section"].get("name", "No Name")
                break

        issues_details[task["name"]] = {
            "description": task_data.get("notes", ""),
            "status": "Completed" if task_data.get("completed") else "Incomplete",
            "assignee": assignee,
            "list": list_name,
        }

    return issues_details

# Run the example flow
# if __name__ == "__main__":


    # assign_user_to_task("task management", "get it done", "David Greiver")
    # move_task_to_section("get it done", "task management", "in progress")
    # print(get_all_issues_details("task management"))
    # get_user_id_by_name("David Greiver")

