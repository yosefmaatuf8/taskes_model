import os
# import subprocess
# import numpy as np
# import torch
# import pandas as pd
# from pydub import AudioSegment
# from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
# from pyannote.core import Segment
# from io import BytesIO
# import requests
# import tiktoken
# import json
# from dotenv import load_dotenv
# from globals import GLOBALS
import datetime
import subprocess
import numpy as np
import json
import torch
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from pydub import AudioSegment
from io import BytesIO
import requests
import tiktoken
import pandas as pd
from globals import GLOBALS


class TranscriptionHandler:
    def __init__(self, mp3_path=None, language=GLOBALS.language, db_path=GLOBALS.db_path):
        self.api_key = GLOBALS.API_KEY
        self.language = language
        self.db = pd.read_csv(db_path)  # DB with embeddings and speaker names
        self.mp3_path = mp3_path
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.audio_helper = Audio()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = GLOBALS.max_tokens
        self.max_audio_duration = 1200

    @staticmethod
    def convert_to_wav(self, path):
        """Convert audio file to WAV format if it's not already WAV."""
        if not path.lower().endswith(".wav"):
            wav_path = "converted_audio.wav"
            subprocess.call(['ffmpeg', '-i', path, wav_path, '-y'])
            return wav_path
        return path

    def segment_embedding(self, segment, path):
        """Extract the embedding for a specific segment."""
        start = segment["start"]
        end = segment["end"]
        clip = Segment(start, end)
        waveform, sample_rate = self.audio_helper.crop(path, clip)
        return self.embedding_model(waveform[None])

    def transcribe_audio_with_whisper(self, audio):
        """Transcribe a single audio segment using OpenAI Whisper with timestamps."""
        buffer = BytesIO()
        audio.export(buffer, format="mp3")
        buffer.seek(0)

        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"file": ("audio.mp3", buffer, "audio/mp3")}
        data = {
            "model": "whisper-1",
            "language": self.language,
            "response_format": "verbose_json"
        }

        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error in transcription: {response.status_code}, {response.text}")

    def split_audio_if_needed(self, audio):
        """Split audio into smaller chunks if it exceeds the maximum duration."""
        audio_duration = len(audio) / 1000  # Convert from milliseconds to seconds
        if audio_duration > self.max_audio_duration:
            print(
                f"Audio duration is {audio_duration}s, exceeding the limit of {self.max_audio_duration}s. Splitting...")
            chunks = []
            for start_time in range(0, int(audio_duration), self.max_audio_duration):
                end_time = min(start_time + self.max_audio_duration, int(audio_duration))
                chunk = audio[start_time * 1000:end_time * 1000]
                chunks.append(chunk)
            return chunks
        else:
            return [audio]

    def get_speaker_embeddings_from_db(self):
        """Retrieve embeddings from the DB for each speaker."""
        embeddings = []
        for _, row in self.db.iterrows():
            embeddings.append(np.array(row['embedding']))
        return np.vstack(embeddings)

    def find_closest_speaker(self, embedding, db_embeddings):
        """Find the closest speaker by comparing embeddings."""
        similarities = np.dot(db_embeddings, embedding.T)
        closest_speaker_index = np.argmax(similarities)
        return self.db.loc[closest_speaker_index, 'name']

    def process_transcription_with_speakers(self, json_transcription):
        """Process the transcription and match speakers using embeddings."""
        speaker_embeddings = self.get_speaker_embeddings_from_db()
        conversation = []

        for segment in json_transcription['segments']:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]

            # Convert the audio segment to waveform and extract its embedding
            embedding = self.segment_embedding(segment, self.mp3_path)

            # Find closest speaker in the DB
            speaker_name = self.find_closest_speaker(embedding, speaker_embeddings)

            # Add speaker name to the transcription
            conversation.append({
                "start": start_time,
                "end": end_time,
                "text": text,
                "speaker": speaker_name
            })

        return conversation

    def save_transcription_to_json(self, transcription, output_file):
        """Save the transcription to a JSON file."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transcription, f, ensure_ascii=False, indent=4)

    def run(self, output_dir=GLOBALS.output_path, output_file=None):
        """Main method to execute the transcription and speaker identification."""
        # Step 1: Load audio
        audio = AudioSegment.from_file(self.mp3_path, format="mp3")

        # Step 2: Split audio if needed
        audio_chunks = self.split_audio_if_needed(audio)

        # Step 3: Perform transcription and speaker identification on each chunk
        full_transcription = []

        for chunk in audio_chunks:
            json_transcription = self.transcribe_audio_with_whisper(chunk)
            transcription_with_speakers = self.process_transcription_with_speakers(json_transcription)
            full_transcription.extend(transcription_with_speakers)

        # Step 4: Save the final result to a JSON file
        today = datetime.datetime.today().strftime('%d/%m/%y')
        if output_file is None:
            output_file = f"transcription_daily_{today}.json"
        output_path = os.path.join(output_dir, output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.save_transcription_to_json(full_transcription, output_path)
        print(f"Transcription saved to {output_file}")
        return output_path

