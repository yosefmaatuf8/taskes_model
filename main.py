import os
import threading
import time
from transcription_handler import TranscriptionHandler
from stream_recorder import StreamRecorder
from daily_summary import DailySummary
from extract_tasks import ExtractTasks
from trello_api import TrelloAPI


class MeetingProcessor:
    def __init__(self, output_dir="recordings", language="english"):
        """
        Meeting management class that includes:
        - Real-time audio recording
        - Sending audio chunks for transcription during the meeting
        - Keeping track of speaker identification
        - Managing summary and tasks at the end of the meeting
        """
        self.output_dir = output_dir
        self.language = language
        self.recorder = StreamRecorder(output_dir=output_dir)
        self.transcription_handler = TranscriptionHandler(language=language)
        self.daily_summary = DailySummary()
        self.extract_tasks = ExtractTasks()
        self.trello_api = TrelloAPI()
        self.speaker_embeddings = {}  # Storing speaker identification
        self.transcriptions = []  # Storing transcriptions
        self.is_running = False

    def start_meeting(self):
        """
        Start the meeting – begins recording and monitoring files.
        """
        self.is_running = True
        self.recorder.start_recording()
        monitoring_thread = threading.Thread(target=self.monitor_audio_chunks, daemon=True)
        monitoring_thread.start()
        print("📢 The meeting has started!")

    def monitor_audio_chunks(self):
        """
        Detects new audio chunks and sends them for transcription.
        """
        processed_files = set()
        while self.is_running:
            for filename in os.listdir(self.output_dir):
                if filename.startswith("chunk_") and filename.endswith(".wav"):
                    filepath = os.path.join(self.output_dir, filename)
                    if filepath not in processed_files:
                        self.process_audio_chunk(filepath)
                        processed_files.add(filepath)
            time.sleep(2)  # Check every 2 seconds

    def process_audio_chunk(self, filepath):
        """
        Sends a new file for transcription, identifies speakers, and stores the results.
        """
        print(f"🎙️ Processing new chunk: {filepath}")
        transcript_data = self.transcription_handler.transcribe_audio(filepath)

        # Store transcription
        self.transcriptions.append(transcript_data)

        # Update speaker identification
        self.update_speaker_identification(transcript_data)

    def update_speaker_identification(self, transcript_data):
        """
        Identifies speakers and maintains speaker consistency throughout the meeting.
        """
        for segment in transcript_data:
            speaker_id = segment["speaker"]
            if speaker_id not in self.speaker_embeddings:
                self.speaker_embeddings[speaker_id] = self.transcription_handler.extract_speaker_embedding(
                    segment["audio"])

    def end_meeting(self):
        """
        Ends the meeting, processes all data, and sends summary and tasks.
        """
        self.is_running = False
        self.recorder.setup_new_meeting()

        print("📜 Processing daily summary...")
        summary = self.daily_summary.generate_summary(self.transcriptions)
        print("✅ Identifying tasks...")
        tasks = self.extract_tasks.process_transcriptions(self.transcriptions)

        print("📌 Sending to Trello...")
        self.trello_api.create_trello_cards(tasks)

        print("🎤 The meeting has ended. All data has been saved!")


# Example usage:
if __name__ == "__main__":
    meeting = MeetingProcessor()
    meeting.start_meeting()
    time.sleep(60)  # One-minute meeting for demonstration
    meeting.end_meeting()
