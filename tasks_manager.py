import os
from daily_summary import DailySummary
from extract_tasks import ExtractTasks
from trello_api import TrelloAPI
from globals import GLOBALS
from post_transcription_data_processor import PostTranscriptionDataProcessor, TranscriptClassifier
from transcription_handler_1 import TranscriptionHandler
from init_project import InitProject
from db_manager import DBManager
import json


class TasksManager:
    def __init__(self,wav_path = None,
                 sender_email =GLOBALS.sender_email,
                 sender_password = GLOBALS.sender_password):
        """Initialize the Main class and prepare transcription handling."""
        self.data_for_summary = None
        self.tasks_to_update = None
        self.transcription_txt = None
        self.transcriptClassifier = None
        self.wav_path = wav_path
        self.db_manager = DBManager()
        self.transcription_handler = None
        self.transcription_for_ask_model = None
        self.transcription_json = None  # To store the transcription content
        self.extract_tasks = None  # To process transcription tasks
        self.tasks = None
        self.trello_api_employees = TrelloAPI(GLOBALS.bord_id_employees)
        # self.to_emails = to_emails  # Load email recipients from globals
        self.summary_model = DailySummary(sender_email, sender_password)  # Initialize email sender
        self.summary = None  # Placeholder for the generated summary
        self.transcription_handler = TranscriptionHandler(wav_path)
        try:
            GLOBALS.users_name_trello = self.trello_api_employees.get_usernames()
            GLOBALS.id_users_name_trello = self.trello_api_employees.get_id_and_usernames()
            print(GLOBALS.id_users_name_trello )
            GLOBALS.list_tasks = self.trello_api_employees.get_all_card_details()
        except Exception as e:
            print(f"Error loading Trello data: {e}")

        if not os.path.exists (self.db_manager.db_tasks_path) or not os.path.exists (self.db_manager.db_users_path):
            init_project = InitProject()
            init_project.run()
        self.trello_api_manager = TrelloAPI()



    def process_transcription(self,wav_path):
        self.transcription_handler = TranscriptionHandler(wav_path)
        self.transcription_json, self.transcription_txt = self.transcription_handler.run_all_file()
        return self.transcription_json


    def read_transcription(self):
        """Read the transcription from a file."""
        try:
            with open(self.transcription_json, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Transcription file not found.")


    def process_extract(self):
        """Process the transcription and generate tasks."""
        if not self.transcription_for_ask_model:
            print("Error: No transcription text available for processing.")
            return None
        self.tasks_to_update, self.data_for_summary = TranscriptClassifier(self.transcription_for_ask_model, self.trello_api_employees)

        # Initialize the transcription handler and generate tasks
        self.extract_tasks = ExtractTasks(self.transcription_for_ask_model, self.trello_api_employees)
        self.tasks = self.extract_tasks.generate_tasks()
        print("Tasks generated successfully.")
        return self.tasks

    def daily_summary(self):
        """Generate and send a daily summary via email."""
        if not self.transcription_json:
            print("Error: No transcription text available to process.")
            return

        print("Processing and sending the daily summary...")
        self.summary_model.process_and_notify(self.data_for_summary,self.tasks)
        print("Daily summary sent successfully.")

    def extract_and_run_tasks(self):
        """
        Extract JSON tasks from the provided input, validate them, and execute Trello tasks.
        """
        all_tasks = []  # Store all valid tasks

        # Iterate over the tasks and process each one
        for task in self.tasks:
            if isinstance(task, str) and task.strip():  # Ensure the string is not empty or whitespace
                try:
                    str_task = task.split('&', 1)[1]
                    print("this is parsed_task -- ",str_task)

                except ValueError:
                    print("The expected separator '&' was not found in the model output.")
                    continue

                try:
                    # Attempt to parse the string as JSON
                    parsed_task = json.loads(str_task)

                    # Ensure the parsed data is a list of dictionaries
                    if isinstance(parsed_task, list) and all(isinstance(t, dict) for t in parsed_task):
                        all_tasks.extend(parsed_task)  # Add valid tasks to the main list
                    else:
                        print(f"Skipping invalid task format: {parsed_task}")
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON: {str_task}, Error: {e}")
            else:
                print(f"Skipping non-string or empty task: {task}")

        # If there are valid tasks, run them
        if all_tasks:
            self.run_trello_tasks(all_tasks)
        else:
            print("No valid tasks found.")

    def run_trello_tasks(self, tasks):
        """
        Run Trello tasks based on a valid list of tasks.
        Handles nested lists, strings, and dictionaries.
        """
        # Map available functions to Trello API methods
        functions_mapping = {func_name: getattr(self.trello_api_employees, func_name)
                             for func_name in GLOBALS.functions_dict.keys()}

        # Ensure tasks is a list, handle single dictionary or string as input
        if isinstance(tasks, dict):  # Single task as dictionary
            tasks = [tasks]
        elif isinstance(tasks, str):  # Single task as JSON string
            try:
                tasks = json.loads(tasks)
                if not isinstance(tasks, list):  # Ensure it's a list
                    tasks = [tasks]
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON: {tasks}, Error: {e}")
                return

        # Process each task in the list
        for task in tasks:
            print("type task = ", type(task))  # Debugging

            if isinstance(task, list):  # Handle nested lists
                self.run_trello_tasks(task)
                continue

            if not isinstance(task, dict):  # Skip non-dictionary tasks
                print(f"Skipping invalid task: {task}")
                continue

            # Ensure the task has required fields
            if 'function' not in task or 'params' not in task:
                print(f"Skipping incomplete task: {task}")
                continue

            # Execute the task if the function is available
            try:
                if task['function'] in functions_mapping:
                    params = task.get('params', [])
                    print(f"Executing task: {task['function']} with params: {params}")
                    # functions_mapping[task['function']](*params)
                else:
                    print(f"Unknown function: {task['function']}")
            except Exception as e:
                print(f"Error executing task {task}: {e}")

    def ran_for_chunks(self, start_time=0, end_time=0):
        if not os.path.exists(self.wav_path):
            print("Aborting: No wav_path found.")
            return
        self.transcription_for_ask_model = self.transcription_handler.run(start_time, end_time)
        self.tasks = self.process_extract()
        print(self.tasks)
        ask_model = None # create funk for ask model obout tasks
        pass

    def stop_transcription(self):
        self.transcription_txt = self.transcription_handler.output_path_txt
        self.transcription_json = self.transcription_handler.output_path_json
        self.run_all()

    def run_all(self):
        """Run the main workflow."""
        if not os.path.exists(self.wav_path):
            print("Aborting: No wav_path found.")
            return

        print("Processing transcription...")
        self.transcription_json, self.transcription_txt = self.process_transcription(self.wav_path)
        self.read_transcription()  # Read transcription from file


        self.tasks = self.process_extract()  # Process generate tasks

        if self.tasks:
            self.extract_and_run_tasks()

            self.db_manager.generate_updates_from_model(self.tasks_to_update)
            print("Sending daily summary...")
        else:
            print("No tasks generated. Skipping summary.")
        self.daily_summary()  # Send the daily summary


if __name__ == "__main__":
    test = TasksManager()
    # test.run()

