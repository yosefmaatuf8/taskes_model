import json
import tiktoken

from openai import OpenAI
from globals import GLOBALS
from utils.utils import extract_json_from_code_block, split_text
from db_manager import DBManager


class TranscriptionProcessing:
    def __init__(self):
        self.client = OpenAI(api_key=GLOBALS.openai_api_key)
        self.max_tokens = GLOBALS.max_tokens - 500  # Safety margin to prevent response truncation
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.db_manager = DBManager()

    def classify_transcription(self, transcription_data):
        """
        Receives raw transcription and classifies each sentence as Task / Update / General / Irrelevant
        """
        prompt = f"""
        You are a meeting analysis assistant.
        Classify each sentence into one of the following categories:
        - 'Task' if it defines a new task.
        - 'Update' if it is an update to an existing task.
        - 'General' if it is relevant but not a task or update.
        - 'Irrelevant' if it is unrelated.

        Return JSON in the following format:
        ```json
        {{
            "classified_sentences": [
                {{ "sentence": "...", "speaker": "John", "category": "Task"}}
            ]
        }}
        ```
         - **Ensure response is valid JSON**, with no additional text.


        Transcription:
        {transcription_data}
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You classify meeting sentences."},
                      {"role": "user", "content": prompt}],
            max_tokens=4500,
            temperature=0.2
        )
        return extract_json_from_code_block(response.choices[0].message.content)

    def refine_transcription(self, classified_data):
        """
        Receives classified transcription and returns a structured and clear version with task relations and speakers.
        """
        prompt = f"""
        You are a meeting analysis assistant.
        For each sentence classified as Task or Update, refine it into structured information:
        - Indicate who spoke, to whom they were speaking, and the meaning.
        - If it is a new task, specify its status and the relevant person.

        Return JSON in the following format:
        ```json
        {{
            "refined_transcription": [
                {{ "speaker": "John", "target": "Sarah", "text": "We need to start testing", "category": "Task", "related_task": "Testing Phase" }}
            ]
        }}
        ```
         - **Ensure response is valid JSON**, with no additional text.

        Classified Transcription:
        {classified_data}
        
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You refine classified meeting data."},
                      {"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.2
        )


        return extract_json_from_code_block(response.choices[0].message.content)

    def process_transcription(self, transcription_data):
        """
        Processes transcription data:
        1. Splits long transcriptions into chunks based on token limits.
        2. Classifies each chunk into categories (Task, Update, General, Irrelevant).
        3. Refines classified data into structured insights.
        4. Ensures valid JSON output for each step.
        """

        refined_results = []
        max_token = self.max_tokens - 4750
        token_count = len(self.tokenizer.encode(transcription_data))

        # Split text into manageable chunks
        chunks = split_text(self.tokenizer, transcription_data, max_token) if token_count > max_token else [
            transcription_data]

        for chunk in chunks:

            try:
                classified = self.classify_transcription(chunk)

                if not classified:
                    print("❌ Invalid classification response.")
                    continue

                refined_results_classified = []

                classified_chunks = split_text(self.tokenizer, json.dumps(classified, ensure_ascii=False),
                                               max_token)

                for classified_chunk in classified_chunks:
                    refined = self.refine_transcription(classified_chunk)

                    if not refined:
                        print("Invalid refinement response.")
                        continue

                    refined_results_classified.extend(refined.get("refined_transcription", []))
                refined_results.extend(refined_results_classified)
            except Exception as e:
                print(f"❌ Error processing chunk: {e}")
                continue
        print("refined_results", refined_results)
        return {"refined_transcription": refined_results}
