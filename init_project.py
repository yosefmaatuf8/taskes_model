from openai import OpenAI
import json
from globals import GLOBALS


class InitProject:
    def __init__(self):
        """
        Initializes the AI model with API key from GLOBALS.
        """
        self.api_key = GLOBALS.openai_api_key
        self.client = OpenAI(api_key=self.api_key)


    def generate_db_rows(self):
        """
        Uses GPT to generate structured database rows based on Trello data.
        """
        users = GLOBALS.users_name_trello
        tasks = GLOBALS.list_tasks

        prompt = self.build_prompt(users, tasks)
        print(prompt)
        response = self.send_to_gpt(prompt)
        print(response)
        # Parse the JSON response from GPT
        data = json.loads(response)

        return data["rows_users"], data["rows_tasks"]

    def build_prompt(self, users, tasks):
        """
        Constructs a structured prompt to send to the AI model.
        """
        return f"""
        You are an expert in task management systems. Given the following Trello data:

        Users:
        {json.dumps(users, indent=2, ensure_ascii=False)}

        Tasks:
        {json.dumps(tasks, indent=2, ensure_ascii=False)}

        Generate structured database rows for users and tasks. 
        Return a JSON with two keys: "rows_users" and "rows_tasks", each containing a list of dictionaries.
        """

    def send_to_gpt(self, prompt):
        """
        Sends the prompt to OpenAI's GPT model and returns the response.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                      {"role": "user", "content": prompt}],
                max_tokens=int(GLOBALS.max_tokens),
                temperature=0.0
            )

            return response["choices"][0]["message"]["content"]

        except Exception as e:
            return f"An error occurred during task extraction: {str(e)}"
