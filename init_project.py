import tiktoken
from dotenv import find_dotenv, load_dotenv, set_key
from globals import GLOBALS
from openai import OpenAI
from trello_api import TrelloAPI
from db_manager import DBManager
from utils import split_text, extract_json_from_code_block


class InitProject:
    def __init__(self, db_manager=None):
        self.api_key = GLOBALS.openai_api_key
        self.db_manager = db_manager or DBManager()
        self.client = OpenAI(api_key=self.api_key)
        self.max_tokens = int(GLOBALS.max_tokens) - 4500
        self.trello_api = TrelloAPI()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def generate_db_rows(self):
        """
        Uses GPT to generate structured database rows based on Trello data.
        - If tasks are too large, splits them into chunks.
        - Merges partial JSON results correctly.
        - Ensures users are only loaded once.
        - Returns structured 'rows_users' (single call), 'db_status_rows', and 'trello_board'.
        """
        users_str = str(GLOBALS.id_users_name_trello)  # Users remain constant
        tasks_str = str(GLOBALS.list_tasks)

        merged_rows_users_list = None  # Load users only once
        merged_db_status_rows_list = []
        merged_trello_board_dict = {}  # Store merged Trello board as a dict

        # Check if the combined token length exceeds the limit
        if len(users_str) + len(tasks_str) > self.max_tokens:
            task_chunks = split_text(self.tokenizer,tasks_str, self.max_tokens - len(users_str))

            for i, chunk in enumerate(task_chunks):
                print(f"Processing chunk {i + 1}/{len(task_chunks)}...")  # Debugging output

                prompt = self.build_prompt(users_str, chunk)
                print("prompt----", prompt)
                response = self.send_to_gpt(prompt)
                print(response)

                rows_users, db_status_rows, trello_board = self.process_gpt_response(response)

                # Store users only once (from the first chunk)
                if merged_rows_users_list is None:
                    merged_rows_users_list = rows_users

                # Append the additional data from the chunk
                merged_db_status_rows_list.extend(db_status_rows)

                if not isinstance(trello_board, dict):
                    print(
                        f"Unexpected format for trello_board: {type(trello_board)}: {trello_board}. Resetting to empty dict.")
                    trello_board = {}
                for list_name, tasks in trello_board.get("lists", {}).items():
                    if list_name not in merged_trello_board_dict:
                        merged_trello_board_dict[list_name] = []
                    merged_trello_board_dict[list_name].extend(tasks)  # Append tasks to the correct list

        else:
            # If data is small enough, send a single request to GPT
            prompt = self.build_prompt(users_str, tasks_str)
            print("prompt:", prompt)
            response = self.send_to_gpt(prompt)
            print("response: ", response)
            rows_users, db_status_rows, trello_board = self.process_gpt_response(response)

            merged_rows_users_list = rows_users  # Load users only once
            merged_db_status_rows_list.extend(db_status_rows)
            if not isinstance(trello_board, dict):
                print(f"Unexpected format for trello_board: {type(trello_board)}: {trello_board}. Resetting to empty dict.")
                trello_board = {}
            # Merge Trello board data correctly
            for list_name, tasks in trello_board.get("lists", {}).items():
                if list_name not in merged_trello_board_dict:
                    merged_trello_board_dict[list_name] = []
                merged_trello_board_dict[list_name].extend(tasks)

        return merged_rows_users_list, merged_db_status_rows_list, {
            "lists": merged_trello_board_dict}  # Wrap trello_board in "lists"

    @staticmethod
    def process_gpt_response(response):
        parsed_data = extract_json_from_code_block(response)
        if parsed_data:
            return (
                parsed_data.get("rows_users", []),
                parsed_data.get("db_status_rows", []),
                parsed_data.get("trello_board", []),
            )
        else:
            print("Unable to parse JSON from response.")
            print(response)
            return [], [], []

    def build_prompt(self, users, tasks):
        """
        Constructs a structured prompt to send to the AI model.
        Ensures that the AI returns valid JSON output.
        """
        return f"""
        You are an expert in project management and task tracking.
        Your goal is to analyze the given Trello data and structure it into a well-defined database
        for project status tracking, while also generating a corresponding Trello board.
        **Return only valid JSON, no explanations or code.**

        --- **Input Data** ---
        Users (list of usernames and IDs):  
        {users}

        Tasks (dictionary with task details from Trello):  
        {tasks}

        --- **Expected Output** ---
        ```json
        {{
          "db_status_rows": [
            {{
              "topic": "Project Section",
              "progress": "Summarized progress",
              "tasks": [
                {{
                  "id": "Task ID",
                  "name": "Task Name",
                  "status": "Derived from list name or description",
                  "assigned_user_id": "Trello *UserName* (for example "sara39")"
                }}
              ]
            }}
          ],
          "rows_users": [
            {{
              "id": "Trello ID or username",
              "user_name": "Trello display name",
              "hebrew_name": "Extracted Hebrew name",
              "hebrew_name_english": "Extracted English name",
              "embedding": ""
            }}
          ],
          "trello_board": {{
            "lists": {{
              "Project Section": [
                {{
                  "name": "Task Name",
                  "description": "Task description",
                  "assigned_user_id": "Trello User ID"
                }}
              ]
            }}
          }}
        }}
        ```
        - **Extract `hebrew_name` and `hebrew_name_english` properly.**  
        - **Infer `status` from list name and description:**  
        - **Ensure every project section (`topic`) contains a list of relevant tasks.**
        - **Make sure all existing and relevant tasks fall under the corresponding project (`topic`) section.**
        - **Make sure tasks are correctly linked to their respective users by `assigned_user_id`.**
    .  
        - **Return JSON only, no extra text.**
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
                max_tokens=4300,  # the response size
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"An error occurred during task extraction: {str(e)}"
            print(error_msg)
            return ""

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

    def create_trello_board_from_data(self, trello_data, board_id=GLOBALS.bord_id_manager):
        """
        Populates a Trello board with lists and tasks based on structured project data.

        Args:
            trello_data (dict): Dictionary containing Trello lists and their tasks.
            board_id (str): The Trello board ID where the lists and tasks should be created.
        """
        try:
            board = self.trello_api.client.get_board(board_id)

            # Ensure trello_data is a dictionary with the expected structure
            if not isinstance(trello_data, dict) or "lists" not in trello_data:
                print("Invalid Trello board data format. Expected a dictionary with a 'lists' key.")
                return

            # Iterate over project sections (lists)
            for list_name, tasks in trello_data["lists"].items():
                if not isinstance(tasks, list):
                    continue
                try:
                    # Create a new list on the Trello board
                    trello_list = board.add_list(list_name)
                    print(f"Created Trello list: {list_name}")

                    # Iterate over tasks in the list
                    for task in tasks:
                        if not isinstance(task, dict):
                            continue

                        card_name = task.get("name", "Untitled Task")
                        try:

                            description = task.get("description", "")
                            assigned_user_id = task.get("assigned_user_id")

                            # Create Trello card for the task
                            card = trello_list.add_card(card_name, description)
                            print(f"  ➜ Added card: {card_name}")

                            # Assign user to the card if an assigned user ID exists
                            if assigned_user_id:
                                card.assign(assigned_user_id)
                                print(f"    ✓ Assigned to user: {assigned_user_id}")

                        except Exception as e:
                            print(f"Error creating card '{card_name}' in list '{list_name}': {e}")

                except Exception as e:
                    print(f"Error creating Trello list '{list_name}': {e}")

        except Exception as e:
            print(f"Failed to fetch Trello board with ID {board_id}: {e}")

    def run(self):
        rows_users, status_rows, trello_data = self.generate_db_rows()

        # board_id = self.create_new_trello_board("status_project_1")
        board_id = "67b5d782eb0035bdd4bb8595"
        if board_id:
            env_file = find_dotenv()
            load_dotenv(env_file)
            set_key(env_file,"BORD_ID_MANAGER", board_id)

            self.trello_api.board_id = board_id
            self.create_trello_board_from_data(trello_data, board_id)
        else:
            print("Failed to create Trello board. Exiting.")
            return
        self.db_manager.create_db(rows_users, status_rows)
