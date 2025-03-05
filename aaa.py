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
        sys.stderr.write(f"Execution for language '{language}' is not supported.\n")
        return '0', f"Language {language} not supported."

    code = clean_code(code)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_filename = tmp_file.name

    try:
        # Print the code (optional, can be removed if not needed)
        print("\n--- Generated Code ---\n")
        print(code)
        print("\n--- Executing Code ---\n")
        result = subprocess.run([sys.executable, tmp_filename],
                                capture_output=True, text=True, check=True)
        print("---- Output ----")
        print(result.stdout)
        if result.stderr:
            print("---- Errors ----")
            print(result.stderr, file=sys.stderr)
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Error during code execution:\n")
        sys.stderr.write(e.stderr)
        return '0', e.stderr
    finally:
        os.remove(tmp_filename)


def file_to_string(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        sys.stderr.write(f"Error: File {file_path} not found.\n")
        return ""


def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
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
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
    else:
        embeddings = np.empty((0, d))

    embeddings = np.vstack([embeddings, embedding])
    np.save(embedding_filename, embeddings)


def is_similar_embedding(embedding, threshold=0.85, embedding_filename="embeddings.npy"):
    if not os.path.exists(embedding_filename):
        return False  # No embeddings have been stored yet

    embeddings = np.load(embedding_filename)
    if embeddings.size == 0:
        return False  # No stored embeddings

    embedding = np.array(embedding).reshape(1, -1)
    similarities = cosine_similarity(embedding, embeddings)[0]
    return any(sim >= threshold for sim in similarities)


message_1 = f'''
write a script that recieves a task from the meating summerization, and writes functions to update the group board at {management_tool}
through the {management_tool} api. the task will come in a certain format, there is no need to stick to it, rather take the information from the task and find the most apropriate way to update the task at {management_tool}.  
dont assume that just because the task comes with certain feilds that they actualy exist in the {management_tool} api.
there is already a workin .env file with the variables: {trello_env_variables}, and the {management_tool} library is instaled. 
make sure the code should cetch any error that might happen and print an informative message and return the message as well. 
just to emphasize dont wrap the script with code fences or enything else, the code needs to be ready to run.
write the code so that the main function will print 1 if the script seccsessfuly updated the task and 0 if not. make sure the code prints nothing else. write error messages in the sys.stderr.write.
'''

dele = '''look at the api to write functions that suit the tasks best. this is the {management_tool} api: {file_to_string(trello_file_name)}.
use the api to write the most appropriate functions for each task.'''

tasks = {
    "Log Meeting Attendance": {
        "description": "Update the Integrated Follow-Up Discussion card with meeting attendance details and add a summary comment.",
        "card_id": "Integrated Follow-Up Discussion",
        "users": ["dgreiver"],
        "add_custom_field": {
            "Attendance": "dgreiver"
        },
        "comment": "Attendance recorded: dgreiver present."
    }
}

for key, value in tasks.items():
    disc = value["description"]
    embedd = generate_embedding(disc)
    # if is_similar_embedding(embedd):
    #     continue  # Skip if a similar task has already been saved

    messages = [{"role": "system", "content": message_1}]
    user = {"role": "user", "content": "the task content is: " + key + " " + str(value)}
    code = generate_response(messages + [user])

    # Try to execute the code and, if it fails, request a fixed version up to 4 times.
    attempts = 0
    output, error_msg = execute_code(code)
    while output.strip() != "1" and attempts < 4:
        fix_prompt = (f"The following code produced the error:\n{error_msg}\n\n"
                      f"Please provide a corrected version of the code that fixes this error. "
                      "Only output the corrected code with no extra commentary:\n\n" + code)
        code = generate_response([{"role": "system", "content": fix_prompt}])
        attempts += 1
        output, error_msg = execute_code(code)

    # Print the final code and output from execution.
    print(code)
    print(output)

    if output.strip() == "1":
        save_task(disc, code)
    else:
        sys.stderr.write("The task did not work for: " + disc + "\n")
