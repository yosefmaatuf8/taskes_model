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
        self.functions_dict = functions_dict
        self.db_path = self.validate_path("DB_PATH")
        self.db_username = 'db_user.csv'
        self.output_path = self.validate_path("OUTPUT_PATH")
        # self.model_diarization = self.validate_env("MODEL_DIARIZATION")
        self.language = "he"
        self.list_tasks = None

        self.stream_url = self.validate_env("STREAM_URL")
        self.bucket_name = self.validate_env("BUCKET_NAME")
        self.aws_access_key_id= self.validate_env("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = self.validate_env("AWS_SECRET_ACCESS_KEY")
        self.chunk_interval = self.validate_env("CHUNK_INTERVAL")
        self.silence_timeout = self.validate_env("SILENCE_TIMEOUT")
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
