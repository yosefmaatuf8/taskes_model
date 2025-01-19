
# from pydub import AudioSegment
import json
from openai import OpenAI
import tiktoken
# import requests
# from io import BytesIO
# from audio_handler import AudioHandler
from task_manager import TaskManager
from globals import GLOBALS


class TranscriptionHandler:
    def __init__(self,transcription):
        self.api_key = GLOBALS.openai_api_key
        self.language = GLOBALS.language
        self.client = OpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        GLOBALS.list_tasks = "pass" #functhein for pull cards from trelo
        self.list_tasks = GLOBALS.list_tasks
        self.max_tokens = GLOBALS.max_tokens
        self.transcription = transcription

        # self.audio_handler = AudioHandler(GLOBALS.huggingface_api_key)
        self.task_manager = TaskManager()



    def parse_names(self, names_string, known_names):
        """Parse the names string into a dictionary of speaker labels."""
        names_dict = {}
        lines = names_string.split("\n")
        for line in lines:
            if line.strip():  # Skip empty lines
                speaker, name = line.split(":", 1)
                speaker = speaker.strip()
                name = name.strip()

                # Only add 'unknown' if the speaker is not in known_names
                if name.lower() == 'unknown' and speaker not in known_names:
                    names_dict[speaker] = name
                elif name.lower() != 'unknown':
                    names_dict[speaker] = name
        return names_dict


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
        tasks_tokens = len(self.tokenizer.encode(json.dumps(list_tasks)))

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
        print("-" * 50)
        return tasks


    def extract_tasks(self,transcription):
        """Extract tasks from the summarized transcription."""
        messages = [
            {"role": "system", "content": f"You are managing a team's task organization based on a meeting transcription and a list of existing tasks. Your roles include:\
                    Creating cards for new tasks.\
                    Assigning team members to tasks based on meeting context or explicit mentions.\
                    Moving cards between lists to reflect task progress. There are three modes: to do, in process and done\
                    Leaving comments on cards to provide updates, context, or references to the meeting discussion.\
                    For each action that needs to be performed, you must create a json file that specifies the type of action and the parameters required. If no need any rule return empty string"},
            {"role": "user", "content": f"transcription: {transcription}. existing tasks: {self.list_tasks}"}
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
