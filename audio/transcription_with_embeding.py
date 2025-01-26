# import os
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

import os
import subprocess
import numpy as np
from sklearn.cluster import AgglomerativeClustering
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
from dotenv import load_dotenv
from globals import GLOBALS

class TranscriptionHandler:
    def __init__(self, language=GLOBALS.LANGUAGE, mp3_path=None, db_path=None, num_speakers=2):
        self.api_key = GLOBALS.API_KEY
        self.language = language
        self.num_speakers = num_speakers
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

    def transcribe_audio_with_whisper(self):
        """Transcribe a single audio segment using OpenAI Whisper with timestamps."""
        buffer = BytesIO()
        audio = AudioSegment.from_file(self.mp3_path, format="mp3")
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

    def run(self, output_file="transcription_with_speakers.json"):
        """Main method to execute the transcription and speaker identification."""
        # Step 1: Perform transcription using Whisper
        json_transcription = self.transcribe_audio_with_whisper()

        # Step 2: Process transcription and match speakers
        transcription_with_speakers = self.process_transcription_with_speakers(json_transcription)

        # Step 3: Save the final result to a JSON file
        self.save_transcription_to_json(transcription_with_speakers, output_file)
        print(f"Transcription saved to {output_file}")

