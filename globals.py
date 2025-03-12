import os
from dotenv import find_dotenv, load_dotenv
from db.functions_dict import functions_dict


class Globals:
    def __init__(self):
        load_dotenv(find_dotenv(),override=True)
        # Validate paths only for relevant variables
        # DB
        self.db_path = self.validate_path("DB_PATH")
        self.db_username = 'db_user.csv'
        self.output_path = self.validate_path("OUTPUT_PATH")
        # OpenAI
        self.max_tokens = int(self.validate_env("MAX_TOKENS"))
        self.openai_api_key = self.validate_env("OPENAI_API_KEY")
        self.openai_model_name = self.validate_env("OPENAI_MODEL_NAME")
        self.language = "he"
        # huggingface
        self.huggingface_api_key = self.validate_env("HUGGINGFACE_API_KEY")
       
        #Google
        self.sender_email = self.validate_env("SENDER_EMAIL")
        self.sender_password = self.validate_env("SENDER_PASSWORD")
        self.manager_email = self.validate_env("MANAGER_EMAIL")
        self.calendar_id = self.validate_env("CALENDAR_ID")
        #Trello
        self.bord_id_employees = self.validate_env("BORD_ID_EMPLOYEES")
        self.bord_id_manager = self.validate_env("BORD_ID_MANAGER")
        self.list_tasks = None
        self.users_name_trello = None
        self.id_users_name_trello = None
        self.functions_dict = functions_dict
        #Zoom
        self.client_id = self.validate_env("ZOOM_CLIENT_ID")
        self.client_secret = self.validate_env("ZOOM_CLIENT_SECRET")
        self.account_id = self.validate_env("ZOOM_ACCOUNT_ID")
        self.meeting_id = self.validate_env("MEETING_ID")
        #AWS
        self.stream_url = self.validate_env("STREAM_URL")
        self.stream_key = self.validate_env("STREAM_KEY")
        self.bucket_name = self.validate_env("BUCKET_NAME")
        self.aws_access_key_id= self.validate_env("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = self.validate_env("AWS_SECRET_ACCESS_KEY")
        self.chunk_interval = int(self.validate_env("CHUNK_INTERVAL"))
        self.silence_timeout = int(self.validate_env("SILENCE_TIMEOUT"))

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
