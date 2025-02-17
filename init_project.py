from openai import OpenAI
import json
from globals import GLOBALS
from utils import split_text, extract_json_from_code_block


class InitProject:
    def __init__(self):
        """
        Initializes the AI model with API key from GLOBALS.
        """
        self.api_key = GLOBALS.openai_api_key
        self.client = OpenAI(api_key=self.api_key)
        # Leave a buffer (e.g., 2000 tokens) to avoid hitting the context limit
        self.max_tokens = int(GLOBALS.max_tokens) - 2000

    def generate_db_rows(self):
        """
        Uses GPT to generate structured database rows based on Trello data.
        - If tasks are large, splits them into chunks.
        - Merges the partial JSON results locally in Python.
        - Returns final 'rows_users' and 'rows_tasks' merged from all chunks.
        """
        users_str = str(GLOBALS.id_users_name_trello)
        tasks_str = str(GLOBALS.list_tasks)

        merged_rows_users = []
        merged_rows_tasks = []

        # If the combined length is too big, we chunk the tasks to reduce prompt size
        if len(users_str) + len(tasks_str) > self.max_tokens:
            task_chunks = split_text(tasks_str, self.max_tokens)
            for chunk in task_chunks:
                prompt = self.build_prompt(users_str, chunk)
                response = self.send_to_gpt(prompt)
                parsed_data = extract_json_from_code_block(response)
                if parsed_data:
                    # Extend (merge) the data from this chunk
                    # (In this example, we just overwrite, but you can do a real merge if needed.)
                    merged_rows_users = parsed_data.get("rows_users", [])
                    merged_rows_tasks = parsed_data.get("rows_tasks", [])
                else:
                    print("Unable to parse JSON from response.")
                    print(response)
        else:
            # If small enough, handle everything in a single prompt
            prompt = self.build_prompt(users_str, tasks_str)
            response = self.send_to_gpt(prompt)
            parsed_data = extract_json_from_code_block(response)
            if parsed_data:
                merged_rows_users = parsed_data.get("rows_users", [])
                merged_rows_tasks = parsed_data.get("rows_tasks", [])
            else:
                print("Unable to parse JSON from response.")
                print(response)

        return merged_rows_users, merged_rows_tasks

    def build_prompt(self, users, tasks):
        """
        Constructs a structured prompt to send to the AI model.
        Notice we explicitly list the columns we want for 'rows_users' and 'rows_tasks'.
        """
        return f"""
        You are an expert in task management systems. Given the following Trello data:

        Users (list of usernames):
        {users}

        Tasks (dictionary-like info):
        {tasks}

        Please generate structured database rows for users and tasks. 
        Specifically:

        1) 'rows_users' must be a list of dictionaries, each with these fields:
           - 'id' (the user's Trello ID, if you can infer it or just put the username if not)
           - 'user_name' (the user's name in trello, if you can infer it or just put the username if not)
           - 'hebrew_name' (the user's Hebrew name if known, or blank)
           - 'hebrew_name_english' (the Hebrew name rewritten in English, or blank)
           - 'embedding' (leave as an empty string \"\" for now)

        2) 'rows_tasks' must be a list of dictionaries, each with any relevant fields:
           e.g. 'task_name', 'description', 'list_name', 'assigned_users', etc.

        Return a JSON code block in this exact format:
        ```json
        {{
          \"rows_users\": [...],
          \"rows_tasks\": [...]
        }}
        ```
        No extra text after the code block, just the JSON.
        """

    def send_to_gpt(self, prompt):
        """
        Sends the prompt to OpenAI's GPT model and returns the response.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2024,  # the response size
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"An error occurred during task extraction: {str(e)}"
            print(error_msg)
            return ""
