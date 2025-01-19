import os
from dotenv import find_dotenv, load_dotenv


class Globals:
    def __init__(self):
        load_dotenv(find_dotenv())
        self.sender_email = self.validate_path("SENDER_EMAIL")
        self.sender_password = self.validate_path("SENDER_PASSWORD")
        self.max_tokens = self.validate_path("MAX_TOKENS")
        self.openai_api_key = self.validate_path("OPENAI_API_KEY")
        self.huggingface_api_key = self.validate_path("HUGGINGFACE_API_KEY")
        self.path_transcription = self.validate_path("PATH_TRANSCRIPTION")
        self.to_emails = []
        self.language = "he"
        self.list_tasks = None
    def validate_path(self, var_name):
        value = os.getenv(var_name)
        if not value or not os.path.exists(value):
            print(f"{var_name} is not set or the path does not exist: {value}")
        return value



GLOBALS = Globals()