import os
import sys
import subprocess
import tiktoken
from dotenv import find_dotenv, load_dotenv, set_key
from globals import GLOBALS
from openai import OpenAI
from trello_api.trello_api import TrelloAPI
from db_manager.db_manager import DBManager
from utils.utils import split_text, extract_json_from_code_block


class InitProject:
    def __init__(self, db_manager=None):
        """ Initialize project setup with API clients and configurations """
        self.api_key = GLOBALS.openai_api_key
        self.db_manager = db_manager or DBManager()
        self.client = OpenAI(api_key=self.api_key)
        self.max_tokens = int(GLOBALS.max_tokens) - 4500
        self.trello_api = TrelloAPI()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    @staticmethod
    def setup_environment():
        """ Ensure virtual environment and dependencies are properly set up """
        if not os.path.exists("venv"):
            print("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", "venv"])

        print("Installing dependencies from requirements.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

        print("Environment setup complete. Activate the virtual environment before running the project.")

    def create_new_trello_board(self, board_name):
        """ Create a new Trello board and handle errors gracefully. """
        try:
            new_board = self.trello_api.client.add_board(board_name)

            print("new_board--",new_board.name,new_board.id)
            GLOBALS.trello_nwe_bord_id = new_board.id
            return new_board.id
        except Exception as e:
            print(f"Error creating Trello board: {e}")
            return None

    def generate_db_rows(self):
        """ 
        Generate structured database rows using Trello data and GPT model.
        - Handles splitting large requests.
        - Ensures valid task classifications.
        """
        users_str = str(GLOBALS.id_users_name_trello)
        tasks_str = str(GLOBALS.list_tasks)

        merged_rows_users_list = None
        merged_db_status_rows_list = []
        merged_trello_board_dict = {}

        if len(users_str) + len(tasks_str) > self.max_tokens:
            task_chunks = split_text(self.tokenizer, tasks_str, self.max_tokens - len(users_str))

            for i, chunk in enumerate(task_chunks):
                print(f"Processing chunk {i + 1}/{len(task_chunks)}...")
                prompt = self.build_prompt(users_str, chunk)
                response = self.send_to_gpt(prompt)

                rows_users, db_status_rows, trello_board = self.process_gpt_response(response)

                if merged_rows_users_list is None:
                    merged_rows_users_list = rows_users

                merged_db_status_rows_list.extend(db_status_rows)

                if not isinstance(trello_board, dict):
                    print(f"Unexpected format for trello_board: {type(trello_board)}. Resetting to empty dict.")
                    trello_board = {}
                for list_name, tasks in trello_board.get("lists", {}).items():
                    if list_name not in merged_trello_board_dict:
                        merged_trello_board_dict[list_name] = []
                    merged_trello_board_dict[list_name].extend(tasks)

        else:
            prompt = self.build_prompt(users_str, tasks_str)
            response = self.send_to_gpt(prompt)
            rows_users, db_status_rows, trello_board = self.process_gpt_response(response)

            merged_rows_users_list = rows_users
            merged_db_status_rows_list.extend(db_status_rows)

            if not isinstance(trello_board, dict):
                print(f"Unexpected format for trello_board: {type(trello_board)}. Resetting to empty dict.")
                trello_board = {}
            for list_name, tasks in trello_board.get("lists", {}).items():
                if list_name not in merged_trello_board_dict:
                    merged_trello_board_dict[list_name] = []
                merged_trello_board_dict[list_name].extend(tasks)

        return merged_rows_users_list, merged_db_status_rows_list, {"lists": merged_trello_board_dict}

    @staticmethod
    def process_gpt_response(response):
        """ Process GPT response and extract structured JSON data """
        parsed_data = extract_json_from_code_block(response)
        if parsed_data:
            return (
                parsed_data.get("rows_users", []),
                parsed_data.get("db_status_rows", []),
                parsed_data.get("trello_board", {}),
            )
        else:
            print("Unable to parse JSON from response.")
            return [], [], {}

    def build_prompt(self, users, tasks):
        """ Construct structured prompt for the AI model """
        return f"""
        You are an AI assistant for project management.
        Analyze the given Trello data and structure it into a valid project database.

        --- Input Data ---
        Users:
        {users}

        Tasks:
        {tasks}

        --- Output Format ---
        ```json
        {{
          "rows_users": [
            {{
              "id": "User ID",
              "name": "Display Name",
              "hebrew_name": "",
              "full_name_english": "",
              "embedding": ""
            }}
          ],
          "full_tasks": [
            {{
              "id": "Task ID",
              "topic": "Categorized topic",
              "name": "Task Name",
              "status": "",
              "assigned_user": "",
              "summary": ""
            }}
          ],
          "trello_board": {{
            "lists": {{
              "Project Section": [
                {{
                  "id": "Task ID",
                  "name": "Task Name",
                  "description": "",
                  "assigned_user_id": "User ID"
                }}
              ]
            }}
          }}
        }}
        ```
        Ensure JSON is valid and complete.
        """

    def send_to_gpt(self, prompt):
        """ Send prompt to OpenAI API """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4300,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during task extraction: {e}")
            return ""

    def run(self):
        """ Initialize DB files and populate them with structured data """
        board_id = GLOBALS.bord_id_manager
        if not board_id:
            board_id = self.create_new_trello_board("Manager Board")
            if board_id: 
                env_file = os.getenv("DOTENV_PATH", "global.env")
                load_dotenv(env_file)
                set_key(env_file, "BORD_ID_MANAGER", board_id)
                self.trello_api.board_id = board_id
            else:
                print("Failed to create Trello board. Exiting.")
        else:
            self.trello_api.board_id = board_id
        rows_users = [{"id": "", "name": "", "hebrew_name": "", "full_name_english": "", "embedding": ""}]
        rows_tasks = [{"id": "", "topic": "", "name": "", "status": "", "assigned_user": ""}]
        rows_topics = [{"topic": "", "category": "", "topic_status": "", "tasks_id": "", "prompt_for_summary": ""}] 
        rows_full_data = [{"id": "", "topic": "", "name": "", "status": "", "assigned_user": "", "summary": ""}]
        rows_meetings = [{"meeting_id": "","full_transcription":"", "transcription": "", "topics": "", "meeting_datetime": ""}]
        rows_categories = [{"category": "Key Projects"}, {"category": "Development Environment"}, {"category": "Side & Secondary Tasks"},{"topics": ""}]

        trello_users, trello_rows_full_data, trello_data = self.generate_db_rows()

        if trello_users:
            rows_users = trello_users

        if trello_rows_full_data:
            merged_rows_full_data = [
                {"topic": item.get("topic", ""), "topic_status": item.get("progress", ""), "tasks_id": ""}
                for item in trello_rows_full_data
            ]
            rows_full_data = merged_rows_full_data

        self.db_manager.create_db(
            rows_users=rows_users,
            rows_tasks=rows_tasks,
            rows_topics=rows_topics,
            rows_full_data=rows_full_data,
            rows_meetings=rows_meetings,
            rows_categories=rows_categories
        )
        

        print("Database initialized with structured data.")
