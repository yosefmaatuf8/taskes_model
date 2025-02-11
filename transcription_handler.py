import os
import datetime
import json
import requests
from scipy.spatial.distance import cosine
from pydub import AudioSegment
from io import BytesIO
from pyannote.audio import Model
from pyannote.audio import Inference, Audio
# from pyannote.core import Segment

from dotenv import load_dotenv
import tiktoken
from globals import GLOBALS
import io
import numpy as np
from tqdm import tqdm


class TranscriptionHandler:
    def __init__(self, wav_path=None, language=GLOBALS.language,output_file =None):
        self.names_context = ''
        self.output_file = output_file
        load_dotenv()
        self.openai_api_key = GLOBALS.openai_api_key
        self.huggingface_api_key = GLOBALS.huggingface_api_key
        self.language = language
        self.wav_path = wav_path
        self.inference = self.load_model()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.audio_helper = Audio()
        self.max_tokens = int(GLOBALS.max_tokens)
        self.max_size = 10  # Maximum audio duration per segment in mb
        self.embeddings = {}
        self.speakers = []
        self.speaker_count = 0
        self.threshold_closest = 0.7
        self.threshold_similarity = 0.2
        self.segment_index = 0
        self.chunks = []
        self.full_transcription = []
        self.unknown_segments = []
        self.unknown_embeddings = {}

    def load_model(self):
        """Load the PyAnnotemodel for embeddings."""
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=self.huggingface_api_key)
        return Inference(model, window="whole")


    def compute_similarity(self, audio):
        """
        Splits a 10-second audio into 2-second chunks, computes embeddings, and returns similarity scores.
        :param audio: Audio signal (numpy array).
        :return: Mean similarity score between segments (float) or None if insufficient data.
        """
        sr = 1000  # Sampling at 1000Hz (ensure this is correct)
        segment_length = 2 * sr  # 2 seconds in samples
        num_segments = 3  # 6 seconds split into 3 segments of 2 seconds

        # Ensure there is enough data
        if len(audio) < num_segments * segment_length:
            return None

        embeddings = []

        try:
            for i in range(num_segments):
                start = i * segment_length
                end = (i + 1) * segment_length
                segment = audio[start:end]

                # Convert to a file-like object in memory (BytesIO)
                buffer = BytesIO()
                segment.export(buffer, format="wav")
                buffer.seek(0)

                embedding = self.inference(buffer)
                embeddings.append(embedding)

                # Compute similarity between all possible pairs
            similarities = [
                1 - cosine(embeddings[i], embeddings[j])
                for i in range(num_segments) for j in range(i + 1, num_segments)
            ]

            if similarities:
                print("mean_similarities-", np.mean(similarities))
            return np.mean(similarities) if similarities else None

        except Exception as e:
            print(f"Error in compute_similarity: {e}")
            return None

    @staticmethod
    def get_closest_speaker(segment_embedding, track_embeddings):
        """Compare embeddings and find the closest speaker."""
        closest_speaker = None
        min_distance = float("inf")

        for speaker, track_embedding in track_embeddings.items():
            # Calculate the cosine distance between embeddings
            distance = cosine(segment_embedding, track_embedding)

            if distance < min_distance:
                min_distance = distance
                closest_speaker = speaker

        return closest_speaker, min_distance


    def assign_speaker_with_similarity(self, audio_clip):
        """Assigns a speaker using similarity checks."""
        buffer = BytesIO()
        audio_clip.export(buffer, format="wav")
        buffer.seek(0)
        embedding = self.inference(buffer)

        if not self.embeddings:
            similarity_score = self.compute_similarity(audio_clip)
            if similarity_score and similarity_score > self.threshold_similarity:
                speaker_label = f"speaker_{self.speaker_count}"
                audio_clip[:6000].export(buffer, format="wav")
                buffer.seek(0)
                embedding = self.inference(buffer)
                self.embeddings[speaker_label] = embedding
                print(f"add {speaker_label}")
                self.speakers.append(speaker_label)
                self.speaker_count += 1
                if len(self.unknown_segments) > 0:
                    reclassified_unknowns =  self.reclassify_unknown_speakers()
                    if len(reclassified_unknowns) > 0:
                        for segment in self.full_transcription:
                            segment["speaker"] = reclassified_unknowns.get(segment["speaker"], segment["speaker"])

            else:
                speaker_label = f"unknown_{len(self.unknown_embeddings)}"
                self.unknown_embeddings[speaker_label] = embedding
            return speaker_label
        closest_speaker, min_distance = self.get_closest_speaker(embedding, self.embeddings)

        if min_distance < self.threshold_closest:
            speaker_label = closest_speaker
            return speaker_label  # Return early if a close match is foundelif not similarity_score or similarity_score < self.threshold_similarity:
        # Only calculate similarity if no close match was found
        similarity_score = self.compute_similarity(audio_clip)

        if similarity_score and similarity_score > self.threshold_similarity:
            speaker_label = f"speaker_{self.speaker_count}"
            audio_clip[:6000].export(buffer, format="wav")
            buffer.seek(0)
            embedding = self.inference(buffer)
            self.embeddings[speaker_label] = embedding
            print(f"add {speaker_label}")
            self.speakers.append(speaker_label)
            self.speaker_count += 1
            if len(self.unknown_segments) > 0:
                reclassified_unknowns =  self.reclassify_unknown_speakers()
                if len(reclassified_unknowns) > 0:
                    for segment in self.full_transcription:
                        segment["speaker"] = reclassified_unknowns.get(segment["speaker"], segment["speaker"])
        else:
            speaker_label = f"unknown_{len(self.unknown_embeddings)}"
            self.unknown_embeddings[speaker_label] = embedding
        return speaker_label


    def reclassify_unknown_speakers(self):
        """Reassigns unknown speakers once new speakers are detected."""
        if len(self.unknown_segments) < 1:
            return []
        reclassified_segments = {}
        remaining_unknown_segments = []

        for segment in self.unknown_segments:
            embedding = self.unknown_embeddings[segment["speaker"]]
            closest_speaker, min_distance = self.get_closest_speaker(embedding, self.embeddings)

            if min_distance < self.threshold_closest:
                segment["speaker"] = closest_speaker
                reclassified_segments[segment["speaker"]] = closest_speaker
            else:
                remaining_unknown_segments.append(segment)
        self.unknown_segments = remaining_unknown_segments
        return reclassified_segments

    def split_audio_if_needed(self, audio):
        """Split audio into smaller chunks if it exceeds the maximum file size."""

        # Step 1: Get full audio size
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")  # Save to buffer
        full_audio_size = len(buffer.getvalue())  # Size in bytes
        full_audio_duration = len(audio)  # Duration in milliseconds

        max_size_bytes = self.max_size * 1024 * 1024  # Convert MB to bytes

        # Step 2: Check if the audio needs splitting
        if full_audio_size > max_size_bytes:
            print(
                f"Audio size is {full_audio_size / (1024 * 1024):.2f}MB, exceeding the limit of {self.max_size}MB. Splitting...")
            chunks = []
            num_chunks = (full_audio_size // max_size_bytes) + 1
            chunk_duration = full_audio_duration // num_chunks  # Split evenly based on duration

            for start_time in range(0, full_audio_duration, chunk_duration):
                end_time = min(start_time + chunk_duration, full_audio_duration)
                chunk = audio[start_time:end_time]
                chunks.append(chunk)

            return chunks
        else:
            return [audio]

    def transcribe_audio_with_whisper(self, audio):
        """Transcribe audio using OpenAI Whisper with timestamps."""
        if len(audio) < 400:
            return None
        buffer = BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)

        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        files = {"file": ("audio.wav", buffer, "audio/wav")}
        data = {
            "model": "whisper-1",
            "language": self.language,
            "response_format": "verbose_json",
            "temperature": 0.0,
            "prompt": "This audio is a daily stand-up meeting of tech employees discussing software development, tasks, and progress. They mention code, deployments, bugs, sprints, and engineering challenges."

        }

        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error in transcription: {response.status_code}, {response.text}")


    def process_transcription_with_clustering(self, json_transcription, audio_chunk):

        """Process transcription, assign speaker clusters, and handle unknowns directly."""
        speakers = []
        segments = json_transcription["segments"]

        for segment in segments:
            start_ms = int((segment["start"]) * 1000)  # Start time in milliseconds, added chunk time
            end_ms = int((segment["end"]) * 1000)  # End time in milliseconds, added chunk time
            segment_audio = audio_chunk[start_ms:end_ms]  # Extract the segment audio
            if end_ms - start_ms  < 400:
                speakers.append("Little_segment")
                continue
            speaker_label = self.assign_speaker_with_similarity(segment_audio)
            speakers.append(speaker_label)

        conversation = []
        for i, segment in enumerate(segments):
            conversation.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": speakers[i],
                "index": self.segment_index
            })
            self.segment_index += 1

        return conversation

    def split_text(self, text, max_tokens):
        """Split text into chunks within the token limit."""
        words = text.split()
        chunks, current_chunk = [], []

        for word in words:
            if len(self.tokenizer.encode(" ".join(current_chunk + [word]))) > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def infer_speaker_names(self, transcription):
        """Uses GPT to infer real speaker names."""
        token_count = len(self.tokenizer.encode(transcription))
        known_names = {}
        names_string = self.names_context

        if token_count > self.max_tokens:
            chunks = self.split_text(transcription, self.max_tokens - 700)
            for chunk in chunks:
                names_string = self.find_in_chunk(chunk, names_string)
                known_names.update(self.parse_names(names_string, known_names))
        else:
            names_string = self.find_in_chunk(transcription, names_string)
            known_names.update(self.parse_names(names_string, known_names))
        print("known_names", known_names)
        return known_names

    def find_in_chunk(self, chunk, names_string):
        """Infer speaker names for a chunk using GPT."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You will receive a conversation transcript with multiple speakers (e.g., speaker_0, speaker_1, etc.).\n"
                    "Your task is to infer the real names of the speakers based on the conversation context.\n"
                    "- If you can determine a speaker's name, return it in the format: `speaker_X: Name`.\n"
                    "- If you cannot determine the name, do not include that speaker in the response.\n"
                    "- Ensure that all names are in English only.\n"
                    "- Maintain consistency with previously assigned names.\n"
                    "- Return only the identified speakers in the exact format shown below, with one speaker per line:\n\n"
                    "speaker_0: Yosef\n"
                    "speaker_2: John\n"
                    "speaker_5: Sarah\n\n"
                    "**Do NOT return 'Unknown' or any unidentified speakers. Do NOT return a dictionary, JSON, or any other format!**"
                ),
            },
            {"role": "system", "content": f"{names_string}."},
            {"role": "user", "content": chunk},
        ]

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.openai_api_key}"},
            json={"model": "gpt-4", "messages": messages, "max_tokens": 300, "temperature": 0},
        )
        print("response-", response.json()["choices"][0]["message"]["content"])
        return response.json()["choices"][0]["message"]["content"]

    def parse_names(self, names_string, known_names):
        """Parse names string into a dictionary."""
        names_dict = {}
        lines = names_string.split("\n")
        for line in lines:
            if line.strip():
                if ":" not in line:
                    continue
                speaker, name = line.split(":", 1)
                speaker = speaker.strip()
                name = name.strip()
                if name.lower() == "unknown" and speaker not in known_names:
                    names_dict[speaker] = name
                elif name.lower() != "unknown":
                    names_dict[speaker] = name
        return names_dict

    def prepare_transcription(self, full_transcription_json):
        """Prepare transcription in the format: 'Speaker: Text' for each line."""
        formatted_transcription = []
        for entry in full_transcription_json:
            speaker = entry.get("speaker", "unknown")
            text = entry.get("text", "")
            formatted_transcription.append(f"{speaker}: {text}")
        return "\n".join(formatted_transcription)

    def run(self, start_time=0, end_time=0, output_dir=GLOBALS.output_path):
        """
        Processes a short audio segment with a 10-second overlap for context.
        """
        audio = AudioSegment.from_file(self.wav_path, format="wav")

        # Ensure the start time includes 10 seconds of overlap (but not below 0)
        start_ms = max(0, (start_time - 10) * 1000)
        end_ms = end_time * 1000
        chunk = audio[start_ms:end_ms]

        # Skip processing if chunk is too short for Whisper
        if len(chunk) < 2000:
            print(f"❌ Segment {start_time}-{end_time} is too short, skipping...")
            return

        print(f"🔹 Processing segment {start_time}-{end_time} (Length: {len(chunk) / 1000:.2f}s)")

        # Step 1: Transcribe audio with Whisper
        json_transcription = self.transcribe_audio_with_whisper(chunk)
        if not json_transcription:
            print("❌ json_transcription is None")
            return

        # Step 2: Process transcription with clustering
        transcription_with_clusters = self.process_transcription_with_clustering(json_transcription, chunk)

        # Step 3: Handle unknown speakers and adjust timestamps
        for segment in transcription_with_clusters:
            # If the segment starts too early (within the 10s overlap), remove old duplicates
            if int(segment["start"]) < 10 and start_time > 10:
                if int(segment["end"]) <= 10:
                    continue
                self.full_transcription.pop()

            # Collect unknown speakers for later reclassification
            if segment["speaker"].startswith("unkno"):
                self.unknown_segments.append(segment)

            # Store final transcription result
            self.full_transcription.append(segment)

        # Step 4: Reclassify unknown speakers
        reclassified_unknowns = self.reclassify_unknown_speakers()
        if len(reclassified_unknowns) > 0:
            for segment in self.full_transcription:
                segment["speaker"] = reclassified_unknowns.get(segment["speaker"], segment["speaker"])

        # Step 5: Infer speaker names from context
        text_transcription = " ".join([seg["text"] for seg in self.full_transcription])
        inferred_names = self.infer_speaker_names(text_transcription)
        self.names_context = inferred_names  # Update context with speaker names

        for segment in self.full_transcription:
            speaker_label = segment['speaker']
            speaker_name = inferred_names.get(speaker_label, segment['speaker'])
            segment['speaker'] = speaker_name  # Update transcription with real speaker names

        # Step 6: Save updated transcription
        today = datetime.datetime.today().strftime('%d_%m_%y')

        # Determine output file paths
        if self.output_file:
            if self.output_file.endswith('.json'):
                output_path_json = os.path.join(output_dir, self.output_file)
                output_path_txt = os.path.join(output_dir, self.output_file.replace('.json', '.txt'))
            elif self.output_file.endswith('.txt'):
                output_path_txt = os.path.join(output_dir, self.output_file)
                output_path_json = os.path.join(output_dir, self.output_file.replace('.txt', '.json'))
            else:
                output_path_json = os.path.join(output_dir, f"transcription_{today}.json")
                output_path_txt = os.path.join(output_dir, f"transcription_{today}.txt")
        else:
            output_path_json = os.path.join(output_dir, f"transcription_{today}.json")
            output_path_txt = os.path.join(output_dir, f"transcription_{today}.txt")

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save transcription to JSON
        with open(output_path_json, "w", encoding="utf-8") as f:
            json.dump(self.full_transcription, f, ensure_ascii=False, indent=4)

        print(f"✅ Transcription saved to {output_path_json}")

        # Convert transcription to text format and save
        with open(output_path_json, 'r') as f:
            full_transcription_json = json.load(f)
            full_transcription_txt = self.prepare_transcription(full_transcription_json)

        with open(output_path_txt, "w", encoding="utf-8") as f:
            f.write(full_transcription_txt)

        print(f"✅ Text transcription saved to {output_path_txt}")

        return output_path_json, output_path_txt

    def run_all_file(self,wav_path = None,output_dir=GLOBALS.output_path):
        if wav_path:
            self.wav_path = wav_path
        audio = AudioSegment.from_file(self.wav_path, format="wav")
        audio_chunks = self.split_audio_if_needed(audio)


        # start_time = 0  # Track the start time of each chunk in the original file
        for chunk in tqdm(audio_chunks, desc="processing chunks..."):
            # Step 2: Transcribe audio
            json_transcription = self.transcribe_audio_with_whisper(chunk)
            if not json_transcription:
                continue

            # # Step 3: Adjust timestamps to match the original audio
            # for segment in json_transcription["segments"]:
            #     segment["start"] += start_time
            #     segment["end"] += start_time
            transcription_with_clusters = self.process_transcription_with_clustering(json_transcription, chunk)
            for segment in transcription_with_clusters:
                if segment["speaker"].startswith("unkno"):
                    self.unknown_segments.append(segment)  # Collect unknown segments

                self.full_transcription.append(segment)  # Add segments

                # Collect unknown segments
        # Insert reclassified segments back into the full transcription in their original positions

        reclassified_unknowns = self.reclassify_unknown_speakers()
        if len(reclassified_unknowns) > 0:
            for segment in self.full_transcription:
                segment["speaker"] = reclassified_unknowns.get(segment["speaker"], segment["speaker"])

        # Step 4: Infer speaker names based on the updated transcription
        text_transcription = " ".join([seg["text"] for seg in self.full_transcription])
        inferred_names = self.infer_speaker_names(text_transcription)
        self.names_context = inferred_names  # Update the context with the latest speaker names

        # Step 5: Update transcription with inferred speaker names
        for segment in self.full_transcription:
            speaker_label = segment['speaker']  # This is the initial "Speaker 1", "Speaker 2", etc.
            speaker_name = inferred_names.get(speaker_label, segment['speaker'])  # Retrieve the actual name
            print("speaker_name-",speaker_name)
            segment['speaker'] = speaker_name  # Update the transcription with the correct name

           # Step 6: Save the final transcription with updated speaker names
        today = datetime.datetime.today().strftime('%d_%m_%y')

        if self.output_file:
            # If output_file ends with .json, use it for the JSON output file
            if self.output_file.endswith('.json'):
                output_path_json = os.path.join(output_dir, self.output_file)
                output_path_txt = os.path.join(output_dir, self.output_file.replace('.json',
                                                                                    '.txt'))  # Change the extension for the text file
            elif self.output_file.endswith('.txt'):
                output_path_txt = os.path.join(output_dir, self.output_file)
                output_path_json = os.path.join(output_dir, self.output_file.replace('.txt',
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
            json.dump(self.full_transcription, f, ensure_ascii=False, indent=4)

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
if __name__ == "__main__":
    file = "../meeting/audio1725163871.wav"
    # Convert MP4 to WAV
    # audio = AudioSegment.from_file(f"{file}.mp4", format="mp4")
    # audio.export(f"{file}.wav", format="wav")
    test = TranscriptionHandler(file)
    test.run_all_file()
