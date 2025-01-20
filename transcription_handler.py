
# from pydub import AudioSegment
import json
from openai import OpenAI
import tiktoken
# import requests
# from io import BytesIO
# from audio_handler import AudioHandler
# from audio.task_manager import TaskManager
from globals import GLOBALS


class TranscriptionHandler:
    def __init__(self,transcription,trello_api):
        self.api_key = GLOBALS.openai_api_key
        self.language = GLOBALS.language
        self.client = OpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.trello_api = trello_api
        GLOBALS.list_tasks =self.trello_api.get_all_card_details()
        self.list_tasks = str(GLOBALS.list_tasks)
        self.functions_dict = GLOBALS.functions_dict
        self.max_tokens = int(GLOBALS.max_tokens)
        self.transcription = transcription

        # self.audio_handler = AudioHandler(GLOBALS.huggingface_api_key)
        # self.task_manager = TaskManager()TaskManager



    def split_text(self, text, max_tokens):
        """Split text into chunks within the token limit."""
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(self.tokenizer.encode(" ".join(current_chunk + [word]))) > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def generate_tasks(self):
        """Main function to generate tasks from long transcriptions."""
        # Check token size of transcription
        transcription_tokens = len(self.tokenizer.encode(self.transcription))
        tasks_tokens = len(self.tokenizer.encode(json.dumps(self.list_tasks)))

        tasks = []

        # Split transcription if needed
        if transcription_tokens + tasks_tokens > self.max_tokens:
            print("Transcription exceeds token limit. Splitting and create tasks for chunks...")
            chunks = self.split_text(self.transcription, self.max_tokens - 1200 - tasks_tokens)  # Leave room for prompt tokens

            for i, chunk in enumerate(chunks):
                print(f"Create tasks from chunk {i + 1}/{len(chunks)}...")
                tasks_chunk = self.extract_tasks(chunk)
                print(tasks_chunk)
                tasks.append(tasks_chunk)

        else:
            # Extract tasks from the summarized transcription
            print("Extracting tasks from transcription...")
            tasks = self.extract_tasks(self.transcription)
            print(tasks)
        print("-" * 50)
        return tasks


    def extract_tasks(self,transcription):
        """Extract tasks from the summarized transcription."""
        functions_description = "\n".join(
            [f"{func}: {params}" for func, params in GLOBALS.functions_dict.items()]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are managing a team's task organization based on a meeting transcription and a list of existing tasks. "
                    "Your roles include:\n"
                    "- Creating cards for new tasks.\n"
                    "- Assigning team members to tasks based on meeting context or explicit mentions.\n"
                    "- Moving cards between lists to reflect task progress (to do, in process, done).\n"
                    "- Leaving comments on cards to provide updates, context, or references to the meeting discussion.\n\n"
                    "You must return a list of dictionaries where each dictionary specifies an action to perform. "
                    "Each dictionary must have the following format:\n"
                    "{ \"function\": \"function_name\", \"params\": [\"param1\", \"param2\", \"param3\"] }.\n\n"
                    "The valid functions and their required parameters are:\n"
                    f"{functions_description}\n\n"
                    "If no actions are needed, return an empty list.\n\n"
                    "Ensure the returned string is valid JSON."
                ),
            },
            {
                "role": "user",
                "content": f"Transcription: {transcription}\nExisting tasks: {self.list_tasks}"
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                # max_tokens=300,  # Output tasks only
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred during task extraction: {str(e)}"

    #
    # def summarize_transcription(self, transcription):
    #     messages = [{"role": "user", "content": f"Summarize this meeting transcript: {transcription}"}]
    #     response = self.client.chat.completions.create(model="gpt-4", messages=messages, max_tokens=500)
    #     return response.choices[0].message.content
