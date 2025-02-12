from dotenv import load_dotenv
from openai import OpenAI
import subprocess
import tempfile
import os
import sys
import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), 
)


MAX_TOKENS = 1000
management_tool = "trello"
trello_env_variables = "TRELLO_API_KEY, TRELLO_API_SECRET, TRELLO_API_TOKEN, TRELLO_BOARD_ID."
trello_file_name = "trello_api.text"


def generate_response(messages, model="o3-mini", client=client):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # max_tokens=MAX_TOKENS
        )
        # Extract the assistant's reply from the response
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error generating response:", e)
        return f"An error occurred: {str(e)}"
    

def clean_code(code: str) -> str:
    cleaned = code.strip()

    # Remove triple backticks if present, optionally with a language specifier
    cleaned = re.sub(r"^```(?:python)?\n", "", cleaned)
    cleaned = re.sub(r"\n```$", "", cleaned)

    # Remove triple single or double quotes if present
    cleaned = re.sub(r"^(\"\"\"|''')", "", cleaned)
    cleaned = re.sub(r"(\"\"\"|''')$", "", cleaned)

    # Remove any surrounding single or double quotes
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()

    return cleaned

def execute_code(code, language='python'):
    if language.lower() != 'python':
        print(f"Execution for language '{language}' is not supported.")
        return
    
    code = clean_code(code)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_filename = tmp_file.name

    try:
        print("\n--- Generated Code ---\n")
        print(code)
        print("\n--- Executing Code ---\n")
        result = subprocess.run([sys.executable, tmp_filename], capture_output=True, text=True, check=True)
        print("---- Output ----")
        print(result.stdout)
        if result.stderr:
            print("---- Errors ----")
            print(result.stderr)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error during code execution:")
        print(e.stderr)
        return '0'
    finally:
        os.remove(tmp_filename)


def file_to_string(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return ""
    

def append_to_file(file_path, data):
    # Open the file in append mode. 'a' will create the file if it doesn't exist.
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(data)
        # Optionally, add a newline if you want each entry on a separate line
        file.write('\n')


def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    # Use attribute access to get the embedding
    return response.data[0].embedding


def save_code_to_json(description, code, filename="results.json"):
    embedding = generate_embedding(description)
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):  # Ensure it's a list
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    data.append({
        "description": description,
        "embedding": embedding,
        "code": code
    })
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def is_similar_embedding(embedding, threshold=0.85, filename="results.json"):
    if not os.path.exists(filename):
        return False  # No file means no stored embeddings

    # Load existing data
    with open(filename, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):  # Ensure it's a list
                return False
        except json.JSONDecodeError:
            return False

    # Extract stored embeddings
    stored_embeddings = [entry["embedding"] for entry in data if "embedding" in entry]

    if not stored_embeddings:
        return False  # No embeddings in the file

    # Compute cosine similarities
    embedding = np.array(embedding).reshape(1, -1)  # Ensure correct shape
    stored_embeddings = np.array(stored_embeddings)

    similarities = cosine_similarity(embedding, stored_embeddings)[0]  # Get similarity scores

    # Return True if any similarity is above the threshold
    return any(sim >= threshold for sim in similarities)


message_1 = f'''
write a script that recieves a task from the meating summerization, and writes functions to update the group board at {management_tool}
through the {management_tool} api. because we will try to run the whole output, the output should be only the script/code with no other words. 
there is already a workin .env file with the variables: {trello_env_variables}, and the {management_tool} library is instaled. 
lppk at the api to write functions that suit the tasks best. this is the {management_tool} api: {file_to_string(trello_file_name)}.
use the api to write the most appropriate functions for each task.
make sure the code should cetch any error that might happen and print an informative message and return the message as well. 
just to emphasize dont wrap the script with code fences or enything else, the code needs to be ready to run.
write the code so that the main function will print 1 if the script seccsessfuly updated the task and 0 if not. make sure no other prints are accuring.
'''

tasks = {
  "Project Kickoff Card": {
    "description": "Create a new project kickoff card with a checklist of initial tasks and a due date for project planning.",
    "list": "Project Kickoff",
    "users": ["dgreiver"],
    "due": "2025-03-15T17:00:00",
    "checklist": [
      "Define project scope",
      "Identify team roles",
      "Establish milestones"
    ]
  },
  "Move Card with Comment": {
    "description": "Move an existing idea card from the 'Ideas' list to 'In Progress' and add a comment summarizing the meeting decision.",
    "from_list": "Ideas",
    "to_list": "In Progress",
    "card_id": "Schedule Follow-up Discussion",
    "users": ["dgreiver"],
    "comment": "Moved to In Progress after team discussion and approval."
  },
  "Update Custom Fields": {
    "description": "Update custom fields on a card to reflect new priority and sprint number as decided in the meeting.",
    "card_id": "Integrated Follow-Up Discussion",
    "users": ["dgreiver"],
    "custom_fields": {
      "Priority": "High",
      "Sprint": "3"
    }
  },
  "Archive Completed Task": {
    "description": "Archive the completed product launch checklist and trigger a notification to update the team.",
    "card_id": "Product Launch Checklist",
    "users": ["dgreiver"],
    "archive": True,
    "notification": {
      "message": "Product launch checklist archived as completed.",
      "recipients": ["dgreiver"]
    }
  },
  "Recurring Daily Standup": {
    "description": "Set up a recurring daily card for team standup updates including a checklist of status items.",
    "list": "Reminders",
    "users": ["dgreiver"],
    "recurrence": "daily",
    "checklist_update": {
      "update_frequency": "every morning",
      "items": [
        "Yesterday's progress",
        "Today's plan",
        "Blockers"
      ]
    }
  },
  "Update Meeting Minutes Attachment": {
    "description": "On the Meeting Presentation Slides card, remove an outdated attachment and attach updated meeting minutes from today's discussion.",
    "card_id": "Meeting Presentation Slides",
    "users": ["dgreiver"],
    "remove_attachment": {
      "attachment_name": "Old Meeting Minutes"
    },
    "add_attachment": {
      "attachment_name": "Feb 9 Meeting Minutes",
      "description": "Minutes from the Feb 9 meeting."
    }
  },
  "Label and Member Refresh": {
    "description": "Update the labels on a card and assign the card to dgreiver as per the latest meeting outcomes.",
    "card_id": "Critical Bug Fix with Automated Rollback",
    "users": ["dgreiver"],
    "update_labels": ["Critical", "Bug"],
    "assign_members": ["dgreiver"]
  },
  "Consolidate Comments into Description": {
    "description": "Aggregate all comments from a card and update its description with a summary of the meeting decisions.",
    "card_id": "Post-Meeting Slack Update",
    "users": ["dgreiver"],
    "comment_summary": "Consolidated feedback: next steps clarified and action items confirmed.",
    "new_description": "Updated description with summarized comments from the meeting."
  },
  "Adjust Due Date with Notification": {
    "description": "Change the due date on a card based on the new deadline set in the meeting and notify the responsible member.",
    "card_id": "Schedule Follow-up Discussion",
    "users": ["dgreiver"],
    "update_due_date": "2025-03-10T12:00:00",
    "notification": {
      "message": "Due date updated to March 10, 2025. Please adjust your schedule accordingly.",
      "recipients": ["dgreiver"]
    }
  },
  "Duplicate Card for Sub-Task": {
    "description": "Duplicate the marketing campaign action items card to create a new sub-task, linking it to the original for reference.",
    "source_card_id": "Action: Marketing Campaign Action Items",
    "users": ["dgreiver"],
    "duplicate": True,
    "dependencies": "This sub-task is part of the larger campaign planning effort."
  },
  "Add Meeting Comment": {
    "description": "Add a comment to the Client Meeting Feedback card summarizing new UI change requests from the client.",
    "card_id": "Client Meeting Feedback",
    "users": ["dgreiver"],
    "comment": "Client requested additional UI refinements for improved usability."
  },
  "Add Checklist to API Card": {
    "description": "Append a checklist to the Trello api calls card detailing testing steps for API integration.",
    "card_id": "Trello api calls",
    "users": ["dgreiver"],
    "add_checklist": [
      "Verify API endpoints",
      "Test integration flows",
      "Document API usage"
    ]
  },
  "Reassign Ownership": {
    "description": "Ensure the fix-login-issue card is assigned to dgreiver following the meeting discussion on task responsibilities.",
    "card_id": "fix-login-issue",
    "users": ["dgreiver"],
    "assign_members": ["dgreiver"]
  },
  "Trigger Rollback Workflow": {
    "description": "On the Prioritize Critical Bug Fix card, trigger a rollback procedure if automated tests fail as per meeting instructions.",
    "card_id": "Prioritize Critical Bug Fix",
    "users": ["dgreiver"],
    "test_suite": {
      "id": "TEST_CRITICAL_BUG",
      "trigger": "on_failure"
    },
    "rollback_procedure": {
      "enabled": True,
      "script": "rollback_script.sh",
      "condition": "if tests fail"
    }
  },
  "Weekly Meeting Reminder": {
    "description": "Set up a recurring weekly meeting reminder card with an attached checklist of topics for discussion.",
    "list": "Meeting Tasks",
    "users": ["dgreiver"],
    "recurrence": "weekly",
    "due": "2025-03-05T09:00:00",
    "checklist": [
      "Review action items",
      "Discuss progress",
      "Plan next steps"
    ],
    "notification": {
      "message": "Weekly meeting reminder set for March 5, 2025 at 9:00 AM.",
      "recipients": ["dgreiver"]
    }
  },
  "Log Meeting Attendance": {
    "description": "Update the Integrated Follow-Up Discussion card with meeting attendance details and add a summary comment.",
    "card_id": "Integrated Follow-Up Discussion",
    "users": ["dgreiver"],
    "add_custom_field": {
      "Attendance": "dgreiver"
    },
    "comment": "Attendance recorded: dgreiver present."
  },
  "Update Description with Meeting Summary": {
    "description": "Revise the Meeting Presentation Slides card description to include detailed meeting outcomes and next steps.",
    "card_id": "Meeting Presentation Slides",
    "users": ["dgreiver"],
    "new_description": "Presentation reviewed. Next steps: refine design approach and schedule follow-up discussion."
  },
  "Mark Bug as High Priority": {
    "description": "Update the critical bug fix card to mark it as high priority and adjust labels accordingly.",
    "card_id": "Critical Bug Fix with Automated Rollback",
    "users": ["dgreiver"],
    "update_labels": ["High Priority", "Bug"],
    "custom_fields": {
      "Priority": "High"
    }
  },
  "Schedule Follow-Up Meeting": {
    "description": "Create a new follow-up meeting card with a checklist of discussion topics and a set deadline.",
    "list": "Meeting Tasks",
    "users": ["dgreiver"],
    "due": "2025-03-20T15:00:00",
    "checklist": [
      "Review previous action items",
      "Discuss progress",
      "Plan next steps"
    ]
  },
  "Integration Testing Update": {
    "description": "Update the Zoom integration card with results from recent integration tests and next steps for improvement.",
    "card_id": "Zoom integration",
    "users": ["dgreiver"],
    "new_description": "Integration tests completed successfully. Next steps: refine API calls and update documentation.",
    "custom_fields": {
      "TestStatus": "Passed"
    }
  },
  "Enhance API Documentation Card": {
    "description": "Create a new task to update the API documentation based on feedback from the meeting, ensuring clarity for integration.",
    "list": "To Do",
    "users": ["dgreiver"],
    "due": "2025-03-12T16:00:00",
    "checklist": [
      "Review current documentation",
      "Incorporate meeting feedback",
      "Finalize updated document"
    ]
  }
}


for key, value in tasks.items():
    disc = value["description"]
    embedd = generate_embedding(disc)
    boolean = is_similar_embedding(embedd)
    if not boolean:
        
        messages = [{"role": "system", "content": message_1}]
        user = {"role": "user", "content": "the task content is :" + key + str(value)}

        code = generate_response(messages + [user])
        print(code)
        flag = execute_code(code)
        print(flag)
        if int(flag) == 1:
            save_code_to_json(disc, code)
        else:
            print("the task did not work")
            print(disc)





