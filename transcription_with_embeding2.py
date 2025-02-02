import os
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
from sklearn.cluster import DBSCAN
import pandas as pd
from globals import GLOBALS


class TranscriptionHandler:
    def __init__(self, mp3_path=None, language=GLOBALS.language):
        self.api_key = GLOBALS.API_KEY
        self.language = language
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
    def convert_to_wav(path):
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
        """Transcribe audio using OpenAI Whisper with timestamps."""
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

    def cluster_speakers(self, embeddings):
        """Cluster speaker embeddings using DBSCAN."""
        embeddings = np.vstack(embeddings)
        clustering = DBSCAN(metric="cosine", eps=0.5, min_samples=2).fit(embeddings)
        return clustering.labels_

    def process_transcription_with_clustering(self, json_transcription, audio_path):
        """Process transcription and assign speaker clusters."""
        embeddings = []
        segments = json_transcription["segments"]

        for segment in segments:
            embedding = self.segment_embedding(segment, audio_path)
            embeddings.append(embedding.detach().numpy())

        # Cluster speakers
        speaker_labels = self.cluster_speakers(embeddings)

        # Combine transcription with speaker labels
        conversation = []
        for i, segment in enumerate(segments):
            conversation.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": f"Speaker {speaker_labels[i]}"
            })

        return conversation

    def split_text(self, text, max_tokens):
        """Split text into chunks within the token limit, maximizing the last chunk."""
        words = text.split()
        chunks, current_chunk = [], []

        for word in reversed(words):  # Loop through words in reverse
            if len(self.tokenizer.encode(" ".join([word] + current_chunk))) > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.insert(0, word)

        if current_chunk:  # Add the last chunk (now it's the first in reverse order)
            chunks.append(" ".join(current_chunk))

        return list(reversed(chunks))  # Reverse chunks to restore original order

    def infer_speaker_names(self, transcription, names_context):
        """Infer speaker names from transcription using GPT."""
        token_count = len(self.tokenizer.encode(transcription))
        known_names = {}
        names_string = names_context

        if token_count > self.max_tokens:
            print("Transcription exceeds token limit. Splitting and processing...")
            chunks = self.split_text(transcription, self.max_tokens - 700)

            for chunk in chunks:
                names_string = self.find_in_chunk(chunk, names_string)
                names = self.parse_names(names_string, known_names)
                known_names.update(names)
        else:
            names_string = self.find_in_chunk(transcription, names_string)
            names = self.parse_names(names_string, known_names)
            known_names.update(names)

        return known_names

    def find_in_chunk(self, chunk, names_string):
        """Infer speaker names for a chunk using GPT."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You will receive a conversation transcript separated by speakers. "
                    "Your job is to infer the real names of speakers based on the conversation context. "
                    "If you don't know a name, write 'unknown'. Maintain consistency with previous names."
                ),
            },
            {"role": "system", "content": f"Names so far: {names_string}."},
            {"role": "user", "content": chunk},
        ]

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": "gpt-4", "messages": messages, "max_tokens": 300},
        )
        return response.json()["choices"][0]["message"]["content"]

    def parse_names(self, names_string, known_names):
        """Parse names string into a dictionary."""
        names_dict = {}
        lines = names_string.split("\n")
        for line in lines:
            if line.strip():
                speaker, name = line.split(":", 1)
                speaker = speaker.strip()
                name = name.strip()
                if name.lower() == "unknown" and speaker not in known_names:
                    names_dict[speaker] = name
                elif name.lower() != "unknown":
                    names_dict[speaker] = name
        return names_dict


    def prepare_transcription(self,full_transcription_json):
        """Prepare transcription in the format: 'Speaker: Text' for each line."""
        formatted_transcription = []
        for entry in full_transcription_json:
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            formatted_transcription.append(f"{speaker}: {text}")
        return "\n".join(formatted_transcription)

    def run(self, output_dir=GLOBALS.output_path, output_file=None):
        """Main method to execute transcription and speaker identification."""
        # Step 1: Load and preprocess audio
        audio = AudioSegment.from_file(self.mp3_path, format="mp3")
        audio_chunks = self.split_audio_if_needed(audio)

        full_transcription = []  # Store the complete transcription with speaker labels
        names_context = ""  # This will store the ongoing inferred names

        for chunk in audio_chunks:
            # Step 2: Transcribe audio
            json_transcription = self.transcribe_audio_with_whisper(chunk)

            # Step 3: Cluster speakers and process transcription with clusters
            transcription_with_clusters = self.process_transcription_with_clustering(json_transcription, self.mp3_path)
            full_transcription.extend(transcription_with_clusters)

            # Step 4: Infer speaker names based on the updated transcription
            text_transcription = " ".join([seg["text"] for seg in transcription_with_clusters])
            inferred_names = self.infer_speaker_names(text_transcription, names_context)
            names_context = inferred_names  # Update the context with the latest speaker names

            # Step 5: Update transcription with inferred speaker names
            for segment in transcription_with_clusters:
                speaker_label = segment['speaker']  # This is the initial "Speaker 1", "Speaker 2", etc.
                speaker_name = inferred_names.get(speaker_label, 'Unknown')  # Retrieve the actual name
                segment['speaker'] = speaker_name  # Update the transcription with the correct name

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

        return output_path_json,output_path_txt