from dailysummary import DailySummary
from transcription_class import TranscriptionHandler
from globals import GLOBALS


class TasksManager:
    def __init__(self,):
        """Initialize the Main class and prepare transcription handling."""
        self.transcription_text = None  # To store the transcription content
        self.transcription_handler = None  # To process transcription tasks
        self.to_emails = GLOBALS.to_emails  # Load email recipients from globals
        self.summary_model = DailySummary(GLOBALS.sender_email, GLOBALS.sender_password)  # Initialize email sender
        self.summary = None  # Placeholder for the generated summary


    def read_transcription(self):
        """Read the transcription from a file."""
        try:
            with open(GLOBALS.path_transcription, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print("Error: Transcription file not found.")
            return ""

    def process_transcription(self):
        """Process the transcription and generate tasks."""
        if not self.transcription_text:
            print("Error: No transcription text available for processing.")
            return None

        # Initialize the transcription handler and generate tasks
        self.transcription_handler = TranscriptionHandler(self.transcription_text)
        tasks = self.transcription_handler.generate_tasks()
        print("Tasks generated successfully.")
        return tasks

    def dailysummary(self):
        """Generate and send a daily summary via email."""
        if not self.transcription_text:
            print("Error: No transcription text available to process.")
            return

        print("Processing and sending the daily summary...")
        self.summary_model.process_and_notify(self.transcription_text, self.to_emails)
        print("Daily summary sent successfully.")

    def api_trello(self):
        """Placeholder for Trello API integration."""
        pass

    def run(self):
        """Run the main workflow."""
        print("Reading transcription...")
        self.transcription_text = self.read_transcription()  # Read transcription from file

        if not self.transcription_text:
            print("Aborting: No transcription text found.")
            return

        print("Processing transcription...")
        tasks = self.process_transcription()  # Process transcription and generate tasks

        if tasks:
            print("Sending daily summary...")
            self.dailysummary()  # Send the daily summary
        else:
            print("No tasks generated. Skipping summary.")
