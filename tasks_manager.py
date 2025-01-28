import os
from dailysummary import DailySummary
from extract_tasks import ExtractTasks
from trello_api import TrelloAPI
from globals import GLOBALS
from transcription_with_embeding import TranscriptionHandler
import json


class TasksManager:
    def __init__(self,
                 sender_email =GLOBALS.sender_email,
                 sender_password = GLOBALS.sender_password):
        """Initialize the Main class and prepare transcription handling."""
        self.transcription_txt = None
        self.mp3_path = None
        self.audio_handler = None
        self.transcription_handler = None
        self.transcription_json = None  # To store the transcription content
        self.extract_tasks = None  # To process transcription tasks
        self.tasks = None
        self.trello_api = None
        # self.to_emails = to_emails  # Load email recipients from globals
        self.summary_model = DailySummary(sender_email, sender_password)  # Initialize email sender
        self.summary = None  # Placeholder for the generated summary


    def process_transcription(self,mp3_path):
        self.transcription_handler = TranscriptionHandler(mp3_path)
        self.transcription_json = self.transcription_handler.run()
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
        if not self.transcription_json:
            print("Error: No transcription text available for processing.")
            return None

        # Initialize the transcription handler and generate tasks
        self.extract_tasks = ExtractTasks(self.transcription_json, self.trello_api)
        tasks = self.extract_tasks.generate_tasks()
        print("Tasks generated successfully.")
        return tasks

    def dailysummary(self):
        """Generate and send a daily summary via email."""
        if not self.transcription_json:
            print("Error: No transcription text available to process.")
            return

        print("Processing and sending the daily summary...")
        self.summary_model.process_and_notify(self.transcription_json)
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
        functions_mapping = {func_name: getattr(self.trello_api, func_name)
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

    def run(self,mp3_path=None):
        """Run the main workflow."""
        self.trello_api = TrelloAPI()
        if not os.path.exists(mp3_path):
            print("Aborting: No mp3_path found.")
            return

        print("Processing transcription...")
        self.transcription_json, self.transcription_txt = self.process_transcription(mp3_path)
        self.read_transcription()  # Read transcription from file


        self.tasks = self.process_extract()  # Process generate tasks

        if self.tasks:
            self.extract_and_run_tasks()
            print("Sending daily summary...")
        else:
            print("No tasks generated. Skipping summary.")
        self.dailysummary()  # Send the daily summary


if __name__ == "__main__":
    test = TasksManager()
    test.run()

