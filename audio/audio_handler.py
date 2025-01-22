from pydub import AudioSegment
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
from globals import GLOBALS


class AudioHandler:
    def __init__(self, file_path=None, model_diarization=None, max_size_mb=24.5):
        """Initialize the AudioHandler with file path and API credentials."""
        load_dotenv()
        self.api_key = GLOBALS.huggingface_api_key
        self.pipeline = Pipeline.from_pretrained(
            model_diarization or GLOBALS.model_diarization,
            use_auth_token=self.api_key
        )
        self.file_path = file_path or GLOBALS.path_audio
        self.max_size_mb = max_size_mb

    def convert_to_mp3(self):
        """Convert M4A audio file to MP3 format."""
        audio = AudioSegment.from_file(self.file_path, format="m4a")
        self.file_path = self.file_path.replace(".m4a", ".mp3")
        audio.export(self.file_path, format="mp3")
        return self.file_path

    def run_diarization(self):
        """Run speaker diarization on an audio file."""
        diarization = self.pipeline({"uri": "audio", "audio": self.file_path})
        rttm_path = self.file_path.replace(".mp3", ".rttm")
        with open(rttm_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)
        self.rttm_path = rttm_path  # Store as an instance variable
        return rttm_path

    def parse_rttm(self):
        """Parse an RTTM file into a list of speaker segments."""
        segments = []
        if not hasattr(self, 'rttm_path'):
            raise ValueError("RTTM file path is not set. Run 'run_diarization' first.")
        with open(self.rttm_path, "r") as rttm_file:
            for line in rttm_file:
                if not line.startswith("#"):  # Skip comments
                    parts = line.strip().split()
                    speaker = parts[7]
                    start_time = float(parts[3])
                    end_time = start_time + float(parts[4])
                    segments.append((speaker, start_time, end_time))
        return segments

    def split_audio(self):
        """Split audio into smaller parts if it exceeds the size limit."""
        audio = AudioSegment.from_file(self.file_path, format="mp3")
        file_size_mb = len(audio) / (1024 * 1024)
        if file_size_mb <= self.max_size_mb:
            return [audio]

        parts = []
        part_duration = len(audio) // int((file_size_mb // self.max_size_mb + 1))
        for start in range(0, len(audio), part_duration):
            parts.append(audio[start:start + part_duration])
        return parts
