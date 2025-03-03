
from post_transcription_data_processor import TranscriptionProcessing
from extract_and_run_functions_trello import ExtractAndRunFunctions
from transcription_handler import TranscriptionHandler
from daily_summary import DailySummary
from init_project import InitProject
from trello_api import TrelloAPI
from db_manager import DBManager
from globals import GLOBALS
import uuid
import json
import os



class Manger:
    def __init__(self,wav_path = None,output_dir=GLOBALS.output_path, meeting_id = None):
        self.wav_path = wav_path
        self.output_dir = output_dir
        self.meeting_id = meeting_id or str(uuid.uuid4())[:5]
        self.meeting_datetime = None
        self.db_manager = DBManager()

        # self.transcription_handler = TranscriptionHandler(wav_path, output_dir) # for creat transcription with speakers
        self.transcription_json_str = None

        self.post_transcription = TranscriptionProcessing()
        self.edited_transcript = None
        self.updated_topics = [{"updated_topics":'None_updated_topics'}]
        self.updated_for_trello = None

        self.trello_list = None
        self.trello_api_employees = TrelloAPI(GLOBALS.bord_id_employees)
        try:

            GLOBALS.users_name_trello = self.trello_api_employees.get_usernames()
            GLOBALS.id_users_name_trello = self.trello_api_employees.get_id_and_usernames()
            GLOBALS.list_tasks = self.trello_list  # self.trello_api_employees.get_all_card_details()
        except Exception as e:
            print(f"Error loading Trello data: {e}")

        if not os.path.exists (self.db_manager.db_tasks_path) or not os.path.exists (self.db_manager.db_users_path):
            init_project = InitProject()
            init_project.run()
        self.trello_api_manager = TrelloAPI(GLOBALS.bord_id_manager)


        self.summary_model = DailySummary()  # Initialize email sender



    def process_transcription(self,wav_path):
        self.transcription_handler = TranscriptionHandler(wav_path)
        self.transcription_json_str = self.transcription_handler.run_all_file()
        return self.transcription_json_str

    def process_post_transcription(self):
        self.edited_transcript = self.post_transcription.process_transcription(self.transcription_json_str)

    def update_db(self):
       output_db =  self.db_manager.update_project_data(self.edited_transcript, self.meeting_id, self.meeting_datetime)
       self.updated_topics = output_db.get('updated_topics')
       self.updated_for_trello = output_db.get('updated_for_trello')


    def update_trello(self):
        extracted_functions = ExtractAndRunFunctions(self.trello_api_manager)
        try:
             extracted_functions.execute_tasks_directly(self.updated_for_trello)
        except Exception as e:
            print(f"Error executing tasks directly: {e}")




    def daily_summary(self):
        meeting_datetime = self.db_manager.get_meeting_datetime(meeting_id=self.meeting_id)
        if meeting_datetime:
          self.summary_model.process_and_notify(self.edited_transcript, self.updated_topics, meeting_datetime)



    def run(self):
        if not os.path.exists(self.wav_path):
            print("Aborting: No wav_path found.")
            return
        self.process_transcription(self.wav_path)
        if not self.transcription_json_str:
            print("Aborting: No transcription json found.")
            return
        self.process_post_transcription()
        if not self.edited_transcript:
            print("Aborting: No edited transcript.")
            return
        self.update_db()
        if self.updated_for_trello:
            self.update_trello()
        if not self.updated_for_trello:
            print("Aborting: No updated trello data.")

        # self.daily_summary()


    def ran_for_chunks(self, start_time=0, end_time=0):
        if not os.path.exists(self.wav_path):
            print("Aborting: No wav_path found.")
            return
        self.transcription_json_str = self.transcription_handler.run(start_time, end_time)
        print(self.transcription_json_str)


    def stop_transcription(self,meeting_datetime):
        self.meeting_datetime = meeting_datetime
        self.transcription_json_str = self.transcription_handler.transcription_for_ask_model
        self.run()

if __name__ == "__main__":
    test = Manger("/home/mefathim/PycharmProjects/taskes_model_v2/db/output/meeting_20250213_120422/full_meeting.wav")
    path = "/home/mefathim/PycharmProjects/taskes_model_v2/db/output/test.json"
    with open(path, "r", encoding="utf-8") as f:
        test.transcription_json_str = json.dumps(json.load(f), ensure_ascii=False)
    test.meeting_datetime = "2025-02-26 12:30"


    trello_list = str( {
      "Streaming Task": {
        "id": "list_1",
        "cards": [
          {
            "id": "task_1",
            "name": "Streaming has not started yet",
            "status": "To Do",
            "assigned_users": ["Amir"],
            "description": "The task has not been completed yet; streaming needs to be started."
          },
          {
            "id": "task_2",
            "name": "Configure streaming and understand how it works",
            "status": "In Progress",
            "assigned_users": ["David"],
            "description": "David is working on understanding how to set up streaming efficiently."
          }
        ]
      },
      "Internet-Dependent Tasks": {
        "id": "list_2",
        "cards": [
          {
            "id": "task_3",
            "name": "Tasks that require extensive internet usage",
            "status": "To Do",
            "assigned_users": ["David"],
            "description": "Various tasks that require significant internet resources."
          }
        ]
      },
      "Front-End Task": {
        "id": "list_3",
        "cards": [
          {
            "id": "task_4",
            "name": "Improve front-end for better user experience",
            "status": "To Do",
            "assigned_users": ["David"],
            "description": "Enhancements needed for a smoother front-end experience."
          }
        ]
      },
      "Computer Connection Task": {
        "id": "list_4",
        "cards": [
          {
            "id": "task_5",
            "name": "Connect all computers to the project network",
            "status": "Completed",
            "assigned_users": ["David"],
            "description": "David has successfully connected all computers to the network."
          }
        ]
      },
      "AWS Task": {
        "id": "list_5",
        "cards": [
          {
            "id": "task_6",
            "name": "Investigate running the system outside AWS",
            "status": "To Do",
            "assigned_users": ["David"],
            "description": "Need to explore ways to run the system outside the AWS environment."
          }
        ]
      },
      "Script Development": {
        "id": "list_6",
        "cards": [
          {
            "id": "task_7",
            "name": "Develop automation scripts for efficiency",
            "status": "To Do",
            "assigned_users": ["David"],
            "description": "Building automation tools to streamline work processes."
          },
          {
            "id": "task_8",
            "name": "Create scripts for data adaptation",
            "status": "In Progress",
            "assigned_users": ["David"],
            "description": "David is working on scripts to adjust and format data."
          }]

      }
    }
    )
    test.trello_list = trello_list
    test.run()