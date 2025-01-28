import os
from dotenv import find_dotenv, load_dotenv
from functions_dict import functions_dict


class Globals:
    def __init__(self):
        load_dotenv(find_dotenv())
        # Validate paths only for relevant variables
        self.sender_email = self.validate_env("SENDER_EMAIL")
        self.sender_password = self.validate_env("SENDER_PASSWORD")
        self.max_tokens = self.validate_env("MAX_TOKENS")
        self.openai_api_key = self.validate_env("OPENAI_API_KEY")
        self.huggingface_api_key = self.validate_env("HUGGINGFACE_API_KEY")
        self.path_transcription = self.validate_path("PATH_TRANSCRIPTION")  # Path validation
        self.functions_dict = functions_dict
        self.db_path = self.validate_path("DB_PATH")
        self.db_username = 'db_user.csv'
        self.output_path = self.validate_path("OUTPUT_PATH")
        self.model_diarization = self.validate_env("MODEL_DIARIZATION")
        self.num_speakers =self.validate_env("NUM_SPEAKERS")
        self.language = "he"
        self.list_tasks = None

    def validate_env(self, var_name):
        """Validate that the environment variable is set."""
        value = os.getenv(var_name)
        if not value:
            print(f"Error: {var_name} is not set.")
        return value

    def validate_path(self, var_name):
        """Validate that the environment variable is set and points to a valid path."""
        value = os.getenv(var_name)
        if not value:
            print(f"Error: {var_name} is not set.")
        elif not os.path.exists(value):
            print(f"Error: Path does not exist for {var_name}: {value}")
        return value


GLOBALS = Globals()
