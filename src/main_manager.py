import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from post_transcription_data_processor import TranscriptionProcessing
from trello_api.extract_and_run_functions_trello import ExtractAndRunFunctions
from transcription_handler import TranscriptionHandler
from daily_summary.daily_summary import DailySummary
from init_project import InitProject
from trello_api.trello_api import TrelloAPI
from db_manager.db_manager import DBManager
from globals import GLOBALS
import uuid
import json



class Manager:
    def __init__(self,wav_path = None,output_dir=GLOBALS.output_path, meeting_id = None):
        self.wav_path = wav_path
        self.output_dir = output_dir
        self.meeting_id = meeting_id or str(uuid.uuid4())[:5]
        self.meeting_datetime = None
        self.db_manager = DBManager()
        print(GLOBALS.openai_model_name)
       
        self.transcription_handler = TranscriptionHandler(wav_path, output_dir) # for creat transcription with speakers
        self.transcription_json_str = None

        self.post_transcription = TranscriptionProcessing()
        self.edited_transcript = None
        self.updated_topics = [{"updated_topics":'None_updated_topics'}]
        self.updated_for_trello = None
        self.id_users_name_trello = None
        self.trello_list = None
        self.trello_api_employees = TrelloAPI(GLOBALS.bord_id_employees)
        try:

            GLOBALS.users_name_trello = self.trello_api_employees.get_usernames()
            GLOBALS.id_users_name_trello = self.id_users_name_trello or self.trello_api_employees.get_id_and_usernames()
            GLOBALS.list_tasks = self.trello_list or self.trello_api_employees.get_all_card_details()
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
       output_db =  self.db_manager.update_project_data(self.transcription_json_str, self.edited_transcript, self.meeting_id, self.meeting_datetime)
       if output_db:
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
        self.daily_summary()
        if self.updated_for_trello:
            self.update_trello()
        if not self.updated_for_trello:
            print("Aborting: No updated trello data.")



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
    test = Manager("db/tests/full_meeting.wav")
    # path = "db/tests/final_kickoff_meeting_transcription.json"
    # with open(path, "r", encoding="utf-8") as f:
    #    test_data = json.load(f)
    # test.transcription_json_str = str(test_data.get("transcription",[]))
    # test.trello_list = str(test_data.get("trello_board"))
    # test.id_users_name_trello = str(test_data.get("names_trello"))

    test.meeting_datetime = "2025-03-26 00:00"
    test.run()