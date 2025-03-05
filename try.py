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


management_tool = "trello"
trello_env_variables = "TRELLO_API_KEY, TRELLO_API_SECRET, TRELLO_API_TOKEN, TRELLO_BOARD_ID."
trello_file_name = "trello_api.text"


def generate_response(messages, model="o3-mini", client=client):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
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
    

def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    # Use attribute access to get the embedding
    return response.data[0].embedding


def save_task(description, code, meta_filename="tasks.json", embedding_filename="embeddings.npy"):
    # Generate embedding for the description
    embedding = generate_embedding(description)  
    
    if os.path.exists(meta_filename):
        with open(meta_filename, "r", encoding="utf-8") as f:
            try:
                tasks = json.load(f)
                if not isinstance(tasks, list):
                    tasks = []
            except json.JSONDecodeError:
                tasks = []
    else:
        tasks = []
    
    # Append new task metadata
    tasks.append({
        "description": description,
        "code": code
    })
    
    # Save updated metadata
    with open(meta_filename, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=4)
    
    # Determine the dimension of the embedding vector
    embedding = np.array(embedding).reshape(1, -1)
    d = embedding.shape[1]
    
    if os.path.exists(embedding_filename):
        embeddings = np.load(embedding_filename)
        # Ensure embeddings are 2D (in case of a single stored vector)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
    else:
        # Start with an empty array with 0 rows and d columns
        embeddings = np.empty((0, d))
    
    # Append the new embedding as a new row
    embeddings = np.vstack([embeddings, embedding])
    
    # Save the updated embeddings matrix
    np.save(embedding_filename, embeddings)


def is_similar_embedding(embedding, threshold=0.85, embedding_filename="embeddings.npy"):

    if not os.path.exists(embedding_filename):
        return False  # No embeddings have been stored yet
    
    embeddings = np.load(embedding_filename)
    if embeddings.size == 0:
        return False  # No stored embeddings
    
    # Ensure the query embedding has the correct shape
    embedding = np.array(embedding).reshape(1, -1)
    
    # Compute cosine similarities between the query embedding and stored embeddings
    similarities = cosine_similarity(embedding, embeddings)[0]
    
    # Return True if any similarity meets or exceeds the threshold
    return any(sim >= threshold for sim in similarities)


message_1 = f'''
write a script that recieves a task from the meating summerization, and writes functions to update the group board at {management_tool}
through the {management_tool} api. because we will try to run the whole output, the output should be only the script/code with no other words. 
there is already a workin .env file with the variables: {trello_env_variables}, and the {management_tool} library is instaled. 
look at the api to write functions that suit the tasks best. this is the {management_tool} api: {file_to_string(trello_file_name)}.
use the api to write the most appropriate functions for each task.
the task will be in a json format, you dont need to keep all the fields, rather just keep the importent information for writing a good task.
account for the possibilaty that there is already a task or table with the givin name you will use.
make sure the code should cetch any error that might happen and print an informative message and return the message as well. 
just to emphasize dont wrap the script with code fences or enything else, the code needs to be ready to run.
write the code so that the main function will print 1 if the script seccsessfuly updated the task and 0 if not. make sure the code prints nothing else. write error messages in the sys.stderr.write.
if there is an error message it means the code did not work and print 0.
'''

tasks = {
  "Schedule Follow-Up Meeting": {
    "description": "Create a new follow-up meeting card with a checklist of discussion topics and a set deadline.",
    "list": "Meeting Tasks",
    "users": ["dgreiver"],
    "due": "2025-03-20T15:00:00",
    "checklist": [
      "Review previous action items",
      "Discuss progress",
      "Plan next steps"
    ],
    "actions_required": "Create a card in the 'Meeting Tasks' list, add a checklist with topics, assign the card to dgreiver, and set the due date."
  },
  "Label and Member Refresh": {
    "description": "Update the labels on a card and assign the card to dgreiver as per the latest meeting outcomes.",
    "card_id": "Critical Bug Fix with Automated Rollback",
    "users": ["dgreiver"],
    "update_labels": ["Critical", "Bug"],
    "assign_members": ["dgreiver"],
    "actions_required": "Update the card's labels and assign it to the appropriate user based on meeting decisions."
  },
  "Consolidate Comments into Description": {
    "description": "Aggregate all comments from a card and update its description with a summary of the meeting decisions.",
    "card_id": "Post-Meeting Slack Update",
    "users": ["dgreiver"],
    "comment_summary": "Consolidated feedback: next steps clarified and action items confirmed.",
    "new_description": "Updated description with summarized comments from the meeting.",
    "actions_required": "Collect all comments, generate a summary, and update the card's description."
  },
  "Adjust Due Date with Notification": {
    "description": "Change the due date on a card based on the new deadline set in the meeting and notify the responsible member.",
    "card_id": "Schedule Follow-up Discussion",
    "users": ["dgreiver"],
    "update_due_date": "2025-03-10T12:00:00",
    "notification": {
      "message": "Due date updated to March 10, 2025. Please adjust your schedule accordingly.",
      "recipients": ["dgreiver"]
    },
    "actions_required": "Update the card's due date and send a notification to the assigned user."
  },
  "Duplicate Card for Sub-Task": {
    "description": "Duplicate the marketing campaign action items card to create a new sub-task, linking it to the original for reference.",
    "source_card_id": "Action: Marketing Campaign Action Items",
    "users": ["dgreiver"],
    "duplicate": True,
    "dependencies": "This sub-task is part of the larger campaign planning effort.",
    "actions_required": "Duplicate an existing card and link the duplicate to the original to denote dependency."
  },
  "Archive Completed Task": {
    "description": "Archive a completed task card and notify the team about its completion.",
    "card_id": "Product Launch Checklist",
    "users": ["dgreiver"],
    "archive": True,
    "notification": {
      "message": "The product launch checklist has been archived as completed.",
      "recipients": ["dgreiver"]
    },
    "actions_required": "Archive the card and send a notification summarizing the archival action."
  },
  "Add Implementation Checklist": {
    "description": "Add a new checklist to an existing card to track the implementation steps of a feature.",
    "card_id": "Trello api calls",
    "users": ["dgreiver"],
    "add_checklist": [
      "Implement endpoint",
      "Write tests",
      "Update documentation"
    ],
    "actions_required": "Attach a checklist with specific implementation steps to the card."
  },
  "Update Meeting Attachment": {
    "description": "Remove the outdated attachment from the meeting presentation slides card and attach the updated version from today's meeting.",
    "card_id": "Meeting Presentation Slides",
    "users": ["dgreiver"],
    "remove_attachment": {
      "attachment_name": "Old Presentation"
    },
    "add_attachment": {
      "attachment_name": "Updated Presentation",
      "description": "Slides from today's meeting."
    },
    "actions_required": "Remove the old attachment and add the new one to keep the card up to date."
  },

}


for key, value in tasks.items():
    disc = value["actions_required"]
    embedd = generate_embedding(disc)
    boolean = is_similar_embedding(embedd)
    boolean = False
    if not boolean:
        
        messages = [{"role": "system", "content": message_1}]
        user = {"role": "user", "content": "the task content is :" + key + str(value)}

        code = generate_response(messages + [user])
        flag = execute_code(code)
        print(flag)
        if int(flag) == 1:
            save_task(disc, code)
        else:
            print("the task did not work")
            print(disc)





