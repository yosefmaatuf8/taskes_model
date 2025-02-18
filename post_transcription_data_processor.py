import pandas as pd
import os
from db_manager import DBManager
import json
from openai import OpenAI
from globals import GLOBALS

class TranscriptClassifier:
    def __init__(self, transcription_data):
        self.transcription_data = transcription_data
        self.client = OpenAI(api_key=GLOBALS.openai_api_key)

    def classify_sentences(self):
        prompt = f"""
        Classify each sentence in the following transcription into one of these categories:
        - Task
        - Update
        - General

        Return a JSON array where each object contains 'sentence', 'speaker', 'category'.

        Transcription:
        {json.dumps(self.transcription_data)}
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You classify sentences from transcripts."},
                      {"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)

# Example usage:
# classifier = TranscriptClassifier(transcription_data)
# classified_data = classifier.classify_sentences()
# print(classified_data)


class PostTranscriptionDataProcessor:
    def __init__(self, transcription_file, db_manager=DBManager()):
        self.transcription_file = transcription_file
        self.db_manager = db_manager
        self.transcription_data = self.load_transcription()
        self.client = OpenAI(api_key=GLOBALS.openai_api_key)

    def load_transcription(self):
        if not os.path.exists(self.transcription_file):
            raise FileNotFoundError("Transcription file not found.")
        with open(self.transcription_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def process_and_update_tasks(self):
        tasks_data = pd.read_csv(self.db_manager.db_tasks_path)
        prompt = self.build_prompt(tasks_data)
        response = self.send_to_gpt(prompt)
        updated_tasks = json.loads(response)

        # Update or add tasks to the database
        self.db_manager.update_or_add_tasks(updated_tasks)

        return updated_tasks

    def build_prompt(self, tasks_data):
        tasks_json = tasks_data.to_json(orient='records')
        return f"""
        You are an AI assistant for project management. Given the following transcription:
        {json.dumps(self.transcription_data)}

        And the current task data:
        {tasks_json}

        Identify tasks that need to be updated, completed, or added. Return a JSON with a list of updated tasks.
        Each task should include 'id', 'description', 'status', and any other relevant fields.
        """

    def send_to_gpt(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a project management assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.2
        )
        return response.choices[0].message.content

    def generate_project_summary(self, updated_tasks):
        prompt = f"""
        Generate a project summary based on the following updated tasks:
        {json.dumps(updated_tasks, indent=4)}
        """
        response = self.send_to_gpt(prompt)
        return response

    def save_summary(self, summary):
        summary_path = os.path.join(GLOBALS.output_path, "project_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Project summary saved to {summary_path}")

# Usage example:
# processor = PostTranscriptionDataProcessor("path_to_transcription.json", db_manager)
# updated_tasks = processor.process_and_update_tasks()
# summary = processor.generate_project_summary(updated_tasks)
# processor.save_summary(summary)
