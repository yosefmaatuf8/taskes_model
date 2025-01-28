
# from pydub import AudioSegment
import json
from openai import OpenAI
import tiktoken
# import requests
# from io import BytesIO
# from audio_handler import AudioHandler
# from audio.task_manager import TaskManager
from globals import GLOBALS

class ExtractTasks:
    def __init__(self, transcription, trello_api):
        self.api_key = GLOBALS.openai_api_key
        self.language = GLOBALS.language
        self.client = OpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.trello_api = trello_api
        GLOBALS.list_tasks = self.trello_api.get_all_card_details()
        self.list_tasks = str(GLOBALS.list_tasks)
        self.users_names = self.trello_api.get_usernames()
        self.functions_dict = GLOBALS.functions_dict
        self.max_tokens = int(GLOBALS.max_tokens)
        self.transcription = transcription

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
        # Format transcription to include speaker information
        formatted_transcription = self.prepare_transcription()

        # Check token size of transcription
        transcription_tokens = len(self.tokenizer.encode(formatted_transcription))
        tasks_tokens = len(self.tokenizer.encode(json.dumps(self.list_tasks)))

        tasks = []

        # Split transcription if needed
        if transcription_tokens + tasks_tokens > self.max_tokens:
            print("Transcription exceeds token limit. Splitting and create tasks for chunks...")
            chunks = self.split_text(formatted_transcription, self.max_tokens - 1200 - tasks_tokens)  # Leave room for prompt tokens

            for i, chunk in enumerate(chunks):
                print(f"Create tasks from chunk {i + 1}/{len(chunks)}...")
                tasks_chunk = self.extract_tasks(chunk)
                print(tasks_chunk)
                tasks.append(tasks_chunk)

        else:
            # Extract tasks from the summarized transcription
            print("Extracting tasks from transcription...")
            task = self.extract_tasks(formatted_transcription)
            tasks.append(task)
            print(tasks)
        print("-" * 50)
        return tasks

    def extract_tasks(self, formatted_transcription):
        """Extract tasks from the summarized transcription."""
        functions_description = "\n".join(
            [f"{func}: {params}" for func, params in GLOBALS.functions_dict.items()]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are responsible for managing the team's task organization based on a meeting transcription and a list of existing tasks. "
                    "Your roles include:\n"
                    "- Creating cards for new tasks.\n"
                    "- Assigning team members to tasks based on the meeting context or explicit mentions.\n"
                    "- Moving cards between lists to reflect task progress (e.g., 'to do', 'in process', 'done').\n"
                    "- Adding comments to cards to provide updates, context, or references to the meeting discussion.\n\n"
                    "You must return a list of dictionaries, each specifying an action to perform. "
                    "Each dictionary must follow this format:\n"
                    "[{ \"function\": \"function_name\", \"params\": [\"param1\", \"param2\", \"param3\"] }].\n\n"
                    "The valid functions and their required parameters are:\n"
                    f"{functions_description}\n\n"
                    "If no actions are needed, return an empty list.\n\n"
                    "**Important Instructions:**\n"
                    "1. Before returning the list, write the sign `&` on a new line and then write the list on the next line.\n"
                    "2. Do not write anything after the list.\n"
                    "3. Ensure the output is valid JSON and starts immediately after the `&` symbol.\n\n"
                    "The following is the thought process you must follow:\n"
                    "1. Divide the text into sections, and determine, based on your understanding and context, whether each section is related to an update on an existing task, a new task, a dilemma, or an unimportant conversation. Consider the speaker's name and any provided data about it.\n"
                    "2. Go through the text again along with your classification. For each relevant section, match it with an appropriate function. If no function applies, move to the next section.\n"
                    "3. Provide a brief summary of the entire text with your classification for each section.\n"
                    "4. Create a list of all the functions in the required JSON format and include it in the response as follows:\n"
                    "```\n"
                    "&\n"
                    "[{ \"function\": \"function_name\", \"params\": [\"param1\", \"param2\", \"param3\"] }]\n"
                    "```\n"
                    "5. Ensure that the list starts immediately after the `&` symbol, and nothing appears after the list."
                ),
            },
            {
                "role": "user",
                "content": f"Transcription: {formatted_transcription}\nExisting tasks: {self.list_tasks}, Users names and roles: {self.users_names}"
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred during task extraction: {str(e)}"
