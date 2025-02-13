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
        """Main method to execute transcription and speaker identification."""
        # Step 1: Load and preprocess audio
        audio = AudioSegment.from_file(self.wav_path, format="wav")
        audio_chunks = self.split_audio_if_needed(audio)

        full_transcription = []  # Store the complete transcription with speaker labels
        names_context = ""  # This will store the ongoing inferred names

        start_time = 0  # Track the start time of each chunk in the original file
        segment_index = 0

        for chunk in tqdm(audio_chunks, desc="processing chunks..."):
            # Step 2: Transcribe audio
            json_transcription = self.transcribe_audio_with_whisper(chunk)
            if not json_transcription:
                continue

            # Step 3: Adjust timestamps to match the original audio
            for segment in json_transcription["segments"]:
                segment["start"] += start_time
                segment["end"] += start_time

            # Step 4: Cluster speakers and process transcription with adjusted timestamps
            transcription_with_clusters = self.process_transcription_with_clustering(json_transcription, chunk,
                                                                                     start_time, segment_index)
            for segment in transcription_with_clusters:
                if segment["speaker"] == "unknown":
                    self.unknown_segments.append((segment_index, segment))  # Collect unknown segments
                full_transcription.append(segment)  # Add segments
                segment_index += 1

            # Update start_time for the next chunk
            start_time += len(chunk) / 1000  # Convert from milliseconds to seconds
            # print(start_time)
            # Process collected unknown segments, maintaining order
            reclassified_unknowns = self.reclassify_unknown_speakers()

            # Insert reclassified segments back into the full transcription in their original positions
            for index, segment in reclassified_unknowns:
                full_transcription.insert(index, segment)

            # # Step 4: Infer speaker names based on the updated transcription
            # text_transcription = " ".join([seg["text"] for seg in transcription_with_clusters])
            # inferred_names = self.infer_speaker_names(text_transcription, names_context)
            # names_context = inferred_names  # Update the context with the latest speaker names

            # # Step 5: Update transcription with inferred speaker names
            # for segment in transcription_with_clusters:
            #     speaker_label = segment['speaker']  # This is the initial "Speaker 1", "Speaker 2", etc.
            #     speaker_name = inferred_names.get(speaker_label, 'Unknown')  # Retrieve the actual name
            #     segment['speaker'] = speaker_name  # Update the transcription with the correct name

        # Step 6: Save the final transcription with updated speaker names
        today = datetime.datetime.today().strftime('%d_%m_%y')

        if output_file:
            # If output_file ends with .json, use it for the JSON output file
            if output_file.endswith('.json'):
                output_path_json = os.path.join(output_dir, output_file)
                output_path_txt = os.path.join(output_dir, output_file.replace('.json',
                                                                               '.txt'))  # Change the extension for the text file
            elif output_file.endswith('.txt'):
                output_path_txt = os.path.join(output_dir, output_file)
                output_path_json = os.path.join(output_dir, output_file.replace('.txt',
                                                                                '.json'))  # Change the extension for the JSON file
            else:
                # If no extension or a different one, use default names
                output_path_json = os.path.join(output_dir, f"transcription_{today}.json")
                output_path_txt = os.path.join(output_dir, f"transcription_{today}.txt")
        else:
            # If no output_file is provided, use default names for both
            output_path_json = os.path.join(output_dir, f"transcription_{today}.json")
            output_path_txt = os.path.join(output_dir, f"transcription_{today}.txt")

            # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

            # Save the final transcription with all speaker names updated in JSON format
        with open(output_path_json, "w", encoding="utf-8") as f:
            json.dump(full_transcription, f, ensure_ascii=False, indent=4)

        print(f"Transcription saved to {output_path_json}")

        # Read the JSON file and prepare the transcription in text format
        with open(output_path_json, 'r') as f:
            full_transcription_json = json.load(f)
            full_transcription_txt = self.prepare_transcription(full_transcription_json)

        # Save the transcription as a text file
        with open(output_path_txt, "w", encoding="utf-8") as f:
            f.write(full_transcription_txt)

        print(f"Text transcription saved to {output_path_txt}")

        return output_path_json, output_path_txt
