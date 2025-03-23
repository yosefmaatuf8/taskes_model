import os
import datetime
import json
import numpy as np
import requests
from pydub import AudioSegment
from io import BytesIO
from pyannote.audio import Audio
import traceback
import copy
import librosa
from utils.utils import split_text
from dotenv import load_dotenv
import tiktoken
from globals import GLOBALS
from db_manager.db_manager import DBManager
import io
from tqdm import tqdm
from embedding_handler import EmbeddingHandler

class TranscriptionHandler:
    def __init__(self, wav_path=None, output_dir=GLOBALS.output_path, output_file=None, language=GLOBALS.language):
        self.transcription_for_ask_model = None
        self.output_dir = output_dir
        self.openai_model_name = GLOBALS.openai_model_name
        self.names_context = ''
        self.output_file = output_file
        self.db_manager = DBManager()
        self.openai_api_key = GLOBALS.openai_api_key
        self.users_name_trello = self.db_manager.read_users_data()[1]
        self.language = language
        self.wav_path = wav_path
        print(f"Transcribing audio file: {self.wav_path}")
        self.tokenizer = tiktoken.encoding_for_model(self.openai_model_name)
        self.audio_helper = Audio()
        self.max_tokens_response = 1000
        self.max_tokens = GLOBALS.max_tokens - self.max_tokens_response
        self.output_path_json = None
        self.output_path_txt = None
        self.max_size = 20  # Maximum audio duration per segment in mb
        self.speakers = []
        self.least_chunk = 4
        self.chunk_size = GLOBALS.chunk_interval
        self.chunks = []
        self.full_transcription = []
        self.unknown_segments = []
        self.unknown_embeddings = {}
        self.embedding_handler = EmbeddingHandler(self.db_manager, self.reclassify_unknown_speakers)


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


    def remove_silent_parts(self, audio_data, threshold, margin): # Takes audio data and sample rate
        """Removes silent parts from audio data (NumPy array)."""
        try:
            frame_length = 512
            hop_length = frame_length // 4
            rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
            db = librosa.amplitude_to_db(rms + 1e-10)
            silent_frames = np.where(db < np.mean(db) + threshold)[0]
            silent_frames_expanded = []
            for frame in silent_frames:
                silent_frames_expanded.extend(range(max(0, frame - margin // hop_length), min(len(db), frame + margin // hop_length)))
            silent_frames_expanded = sorted(list(set(silent_frames_expanded)))

            mask = np.ones_like(db, dtype=bool)
            mask[silent_frames_expanded] = False

            non_silent_indices = []
            for i in range(len(mask)):
                if mask[i]:
                    start = i * hop_length
                    end = min((i + 1) * hop_length, len(audio_data))
                    non_silent_indices.extend(range(start, end))
            cleaned_audio = audio_data[np.array(non_silent_indices)]

            return cleaned_audio

        except Exception as e:
            print(f"Error processing audio: {e}")
            return None


    def load_and_clean_audio(self, chunk_duration=800, threshold=-10, margin=500):  # Add chunk_duration
        """Loads, cleans audio in chunks, and returns a combined AudioSegment."""
        try:
            audio_segment = AudioSegment.from_file(self.wav_path)
            sr = audio_segment.frame_rate
            total_duration = len(audio_segment)  # in milliseconds
            cleaned_audio_chunks = []

            for start in tqdm(range(0, total_duration, chunk_duration * 1000), desc="Remove silent audio chunks"):
                end = min(start + chunk_duration * 1000, total_duration)
                chunk = audio_segment[start:end]

                samples = np.array(chunk.get_array_of_samples(), dtype=np.int16)
                cleaned_samples = self.remove_silent_parts(samples, threshold, margin)

                if cleaned_samples is not None:
                    cleaned_chunk = AudioSegment(
                        cleaned_samples.tobytes(),
                        frame_rate=sr,
                        sample_width=chunk.sample_width,
                        channels=chunk.channels
                    )
                    cleaned_audio_chunks.append(cleaned_chunk)

            # Concatenate the cleaned chunks back together
            if cleaned_audio_chunks:
                cleaned_audio_segment = cleaned_audio_chunks[0]
                for chunk in cleaned_audio_chunks[1:]:
                    cleaned_audio_segment += chunk
                return cleaned_audio_segment
            else:
                return None

        except Exception as e:
            print(f"Error loading or cleaning audio: {e}")
            return None

    def transcribe_audio_with_whisper(self, audio):
        buffer = BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}
        files = {"file": ("audio.wav", buffer, "audio/wav")}
        data = {
            "model": "whisper-1",
            "language": self.language,
            "response_format": "verbose_json",
            "temperature": 0.0,
        }
        response = requests.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, files=files, data=data)
        return response.json() if response.status_code == 200 else None

    def process_transcription(self, json_transcription, audio_chunk):
        speakers = []
        segments = json_transcription["segments"]
        for segment in segments:
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            segment_audio = audio_chunk[start_ms:end_ms]
            speaker_label = self.embedding_handler.assign_speaker_with_similarity(segment_audio)
            speakers.append(speaker_label)
        conversation = [{"start": seg["start"], "end": seg["end"], "text": seg["text"], "speaker": speakers[i]} for i, seg in enumerate(segments)]
        return conversation

    def reclassify_unknown_speakers(self):
        if len(self.unknown_segments) < 1:
            return []
        if len(self.embedding_handler.unknown_embeddings) < 1:
            return []
        reclassified_segments = {}
        remaining_unknown_segments = []
        for segment in self.full_transcription:
            if segment["speaker"].startswith("unknown"):
                embedding = self.embedding_handler.unknown_embeddings[segment["speaker"]]
                closest_speaker, min_distance = self.embedding_handler.get_closest_speaker(embedding)
                if min_distance < self.embedding_handler.threshold_closest:
                    segment["speaker"] = closest_speaker
                    reclassified_segments[segment["speaker"]] = closest_speaker
                else:
                    remaining_unknown_segments.append(segment)
        if len(reclassified_segments) > 0:
            for segment in self.full_transcription:
                segment["speaker"] = reclassified_segments.get(segment["speaker"], segment["speaker"])
        return reclassified_segments

    def infer_speaker_names(self, transcription):
        """Uses GPT to infer real speaker names."""
        token_count = len(self.tokenizer.encode(transcription))
        known_names = {}
        names_string = self.names_context

        if token_count > self.max_tokens:
            chunks = split_text(self.tokenizer,transcription, self.max_tokens - 1000)
            for chunk in chunks:
                names_string = self.find_in_chunk(chunk, names_string)
                known_names.update(self.parse_names(names_string, known_names))
        else:
            names_string = self.find_in_chunk(transcription, names_string)
            known_names.update(self.parse_names(names_string, known_names))
        # print("known_names", known_names)
        return known_names

    def find_in_chunk(self, chunk, names_string):
        """Infer speaker names for a chunk using GPT."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You will receive a conversation transcript with multiple speakers (e.g., speaker_0, speaker_1, etc.).\n"
                    "Your task is to infer the **real names** of the speakers based on:\n"
                    "- **The conversation context**\n"
                    "- **The response patterns between speakers**\n"
                    "- **The provided Trello board names**\n\n"
                    "### **Key Instructions:**\n"
                    "- **DO NOT assume that a speaker’s name is their identity just because it was mentioned.**\n"
                    "  - Instead, infer speaker identities based on who is **addressed** and who **responds**.\n"
                    "  - Example: If speaker_3 asks, 'Yosef, what did you do?' and speaker_5 responds with details, then **speaker_5 is likely Yosef**.\n\n"
                    "- **Prioritize names from Trello** when assigning speaker identities.\n"
                    "  - The Trello board contains **real employee/user names** that should be used whenever possible.\n"
                    "  - If a speaker's identity strongly matches a Trello name, assign that name.\n\n"
                    "- **If a speaker was already assigned a name, keep it unless strong new evidence suggests a better match.**\n\n"
                    "- **Only return clearly identified speakers.**\n"
                    "  - If a speaker cannot be identified with high confidence, **DO NOT** include them in the response.\n"
                    "  - **DO NOT** return placeholders like 'Unknown' or generic terms.\n\n"
                    "- **Expected Output Format (one speaker per line):**\n"
                    "  speaker_0: Yosef\n"
                    "  speaker_2: John\n"
                    "  speaker_5: Sarah\n\n"
                    "- **If a speaker's identity remains unclear, return the following line separately:**\n"
                    "  'Hi! I'm having trouble identifying speaker_X. Can you help me figure out who it is?'\n\n"
                    "**DO NOT return a dictionary, JSON, or any structured data format! The output should be plain text only.**"
                ),
            },
            {"role": "system",
             "content": f"Previously identified names (these may be updated if a better match is found): {names_string}."},
            {"role": "system",
             "content": f"Trello board users (consider these names when identifying speakers): {self.users_name_trello}."},
            {"role": "user", "content": chunk},  # The actual conversation chunk to analyze
        ]

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.openai_api_key}"},
            json={"model": self.openai_model_name, "messages": messages, "max_tokens": self.max_tokens_response, "temperature": 0},
        )
        if response.status_code != 200:
            print(f"Error in OpenAI response: {response.status_code}, {response.text}")
            return ""

        response_json = response.json()
        if "choices" not in response_json or not response_json["choices"]:
            print(f"Unexpected OpenAI response: {response_json}")
            return ""
        return response_json["choices"][0]["message"]["content"]


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

    def run(self, start_time=0, end_time=0):
        """
        Processes a short audio segment with a 10-second overlap for context.
        """
        try:
            audio = AudioSegment.from_file(self.wav_path, format="wav")

            # Ensure the start time includes 10 seconds of overlap (but not below 0)
            start_ms = max(0, (start_time - self.least_chunk) * 1000)
            end_ms = end_time * 1000
            chunk = audio[start_ms:end_ms]

            # Skip processing if chunk is too short for Whisper
            if len(chunk) < 1000:
                print(f"❌ Segment {start_time}-{end_time} is too short, skipping...")
                return

        except Exception as e:
            print(f"Error in loading audio segment: {e}")
            traceback.print_exc()
            return

        try:
            # Step 1: Transcribe audio with Whisper
            json_transcription = self.transcribe_audio_with_whisper(chunk)
            if not json_transcription:
                print("json_transcription is None")
                return

            # Step 2: Process transcription with clustering
            transcription_with_clusters = self.embedding_handler.process_transcription_with_clustering(json_transcription, chunk)

        except Exception as e:
            print(f"Error in transcription processing: {e}")
            traceback.print_exc()
            return

        try:
            # Step 3: Handle unknown speakers and adjust timestamps
            for segment in transcription_with_clusters:
                # If the segment starts too early (within the 10s overlap), remove old duplicates
                if int(segment["start"]) < self.least_chunk and start_time > self.least_chunk:
                    if int(segment["end"]) < self.least_chunk:
                        continue
                    elif len(self.full_transcription) > 0 and self.full_transcription[-1]["end"] > (
                            self.chunk_size + self.least_chunk) - 1:
                        self.full_transcription.pop()

                # Collect unknown speakers for later reclassification
                if segment["speaker"].startswith("unkno"):
                    self.unknown_segments.append(segment)

                # Store final transcription result
                self.full_transcription.append(segment)

        except Exception as e:
            print(f"Error in handling speakers and timestamps: {e}")
            traceback.print_exc()
            return

        try:
            # Step 5: Infer speaker names from context
            text_transcription = str([
                {"speaker": seg["speaker"], "text": seg["text"]}
                for seg in self.full_transcription
            ])
            inferred_names = self.infer_speaker_names(text_transcription)
            self.names_context = inferred_names  # Update context with speaker names
            self.embedding_handler.embeddings = {
                inferred_names.get(key, key): value
                for key, value in self.embedding_handler.embeddings.items()
            }

            full_transcription_with_names = copy.deepcopy(self.full_transcription)
            for segment in full_transcription_with_names:
                speaker_label = segment['speaker']
                speaker_name = inferred_names.get(speaker_label, segment['speaker'])
                segment['speaker'] = speaker_name  # Update transcription with real speaker names
            self.transcription_for_ask_model = str([
                {"speaker": seg["speaker"], "text": seg["text"]}
                for seg in full_transcription_with_names
            ])


        except Exception as e:
            print(f"Error in inferring speaker names: {e}")
            traceback.print_exc()
            return

        try:
            # Step 6: Save updated transcription
            today = datetime.datetime.today().strftime('%d_%m_%y')

            # Determine output file paths
            if self.output_file:
                if self.output_file.endswith('.json'):
                    output_path_json = os.path.join(self.output_dir, self.output_file)
                    output_path_txt = os.path.join(self.output_dir, self.output_file.replace('.json', '.txt'))
                elif self.output_file.endswith('.txt'):
                    output_path_txt = os.path.join(self.output_dir, self.output_file)
                    output_path_json = os.path.join(self.output_dir, self.output_file.replace('.txt', '.json'))
                else:
                    output_path_json = os.path.join(self.output_dir, f"transcription_{today}.json")
                    output_path_txt = os.path.join(self.output_dir, f"transcription_{today}.txt")
            else:
                output_path_json = os.path.join(self.output_dir, f"transcription_{today}.json")
                output_path_txt = os.path.join(self.output_dir, f"transcription_{today}.txt")

            # Ensure output directory exists
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Save transcription to JSON
            with open(output_path_json, "w", encoding="utf-8") as f:
                json.dump(full_transcription_with_names, f, ensure_ascii=False, indent=4)

            # Convert transcription to text format and save
            with open(output_path_json, 'r') as f:
                full_transcription_json = json.load(f)
                full_transcription_txt = self.prepare_transcription(full_transcription_json)
                print(full_transcription_txt)

            with open(output_path_txt, "w", encoding="utf-8") as f:
                f.write(full_transcription_txt)

        except Exception as e:
            print(f"Error in saving transcription: {e}")
            traceback.print_exc()
            return

        return self.transcription_for_ask_model


    def run_all_file(self, wav_path=None, output_dir=GLOBALS.output_path):
        """
        Process the entire audio file in chunks while maintaining logic consistency with `run()`.
        """
        if wav_path:
            self.wav_path = wav_path
    
        try:
            audio = self.load_and_clean_audio()
            audio.export(f"{self.wav_path}_clean.wav", format="wav")
            audio_chunks = self.split_audio_if_needed(audio)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            traceback.print_exc()
            return None

        for chunk in tqdm(audio_chunks, desc="processing chunks..."):
            if len(chunk) < 1000:
                print(f"❌ Skipping short chunk ({len(chunk)}ms)")
                continue

            try:
                # Step 1: Transcribe audio with Whisper
                json_transcription = self.transcribe_audio_with_whisper(chunk)
                if not json_transcription:
                    print("json_transcription is None")
                    continue

                # Step 2: Process transcription with clustering
                transcription_with_clusters = self.embedding_handler.process_transcription_with_clustering(
                    json_transcription, chunk
                )

            except Exception as e:
                print(f"Error in transcription processing: {e}")
                traceback.print_exc()
                continue

            try:
                # Step 3: Handle unknown speakers and adjust timestamps
                for segment in transcription_with_clusters:
                    if segment["speaker"].startswith("unkno"):
                        self.unknown_segments.append(segment)  # Collect unknown speakers

                    # Store final transcription result
                    self.full_transcription.append(segment)

                # ✅ Reclassify unknown speakers after adding new ones
                reclassified_unknowns = self.reclassify_unknown_speakers()
                if reclassified_unknowns:
                    for segment in self.full_transcription:
                        segment["speaker"] = reclassified_unknowns.get(segment["speaker"], segment["speaker"])

            except Exception as e:
                print(f"Error in handling speakers and timestamps: {e}")
                traceback.print_exc()
                continue

        try:
            # Step 4: Infer speaker names based on full transcription
            text_transcription = str([
                {"speaker": seg["speaker"], "text": seg["text"]}
                for seg in self.full_transcription
            ])
            inferred_names = self.infer_speaker_names(text_transcription)
            self.names_context = inferred_names  # Update context with inferred speaker names
            self.embedding_handler.embeddings = {
                inferred_names.get(key, key): value
                for key, value in self.embedding_handler.embeddings.items()
            }
            if self.db_manager.db_users_path:
                self.db_manager.save_user_embeddings(self.embedding_handler.embeddings)
            # Step 5: Update transcription with inferred speaker names
            full_transcription_with_names = copy.deepcopy(self.full_transcription)
            for segment in full_transcription_with_names:
                speaker_label = segment['speaker']
                speaker_name = inferred_names.get(speaker_label, segment['speaker'])
                segment['speaker'] = speaker_name  # Update with real speaker names
            df_users = self.db_manager.read_db("db_users_path")
            dict_username_name = dict(zip(df_users["name"], df_users["full_name_english"]))
            transcription_json = [
                {"speaker":dict_username_name.get(seg["speaker"],seg["speaker"]), "text": seg["text"]}
                for seg in full_transcription_with_names
            ]
            self.transcription_for_ask_model = str(transcription_json)

        except Exception as e:
            print(f"Error in inferring speaker names: {e}")
            traceback.print_exc()
            return None

        try:
            # Step 6: Save updated transcription
            today = datetime.datetime.today().strftime('%d_%m_%y')

            if self.output_file:
                if self.output_file.endswith('.json'):
                    self.output_path_json = os.path.join(output_dir, self.output_file)
                    self.output_path_txt = os.path.join(output_dir, self.output_file.replace('.json', '.txt'))
                elif self.output_file.endswith('.txt'):
                    self.output_path_txt = os.path.join(output_dir, self.output_file)
                    self.output_path_json = os.path.join(output_dir, self.output_file.replace('.txt', '.json'))
                else:
                    self.output_path_json = os.path.join(output_dir, f"transcription_{today}.json")
                    self.output_path_txt = os.path.join(output_dir, f"transcription_{today}.txt")
            else:
                self.output_path_json = os.path.join(output_dir, f"transcription_{today}.json")
                self.output_path_txt = os.path.join(output_dir, f"transcription_{today}.txt")

            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save to JSON
            with open(self.output_path_json, "w", encoding="utf-8") as f:
                json.dump(transcription_json, f, ensure_ascii=False, indent=4)



            # Read JSON file and convert transcription to text format
            with open(self.output_path_json, 'r') as f:
                full_transcription_json = json.load(f)
                full_transcription_txt = self.prepare_transcription(full_transcription_json)

            # Save text transcription
            with open(self.output_path_txt, "w", encoding="utf-8") as f:
                f.write(full_transcription_txt)

            print(f"✅ Text transcription saved to {self.output_path_txt}")

        except Exception as e:
            print(f"Error in saving transcription: {e}")
            traceback.print_exc()
            return None

        return self.transcription_for_ask_model


if __name__ == "__main__":
    file = "/db/meet_25_12_24_first.wav"
    # Convert MP4 to WAV
    # audio = AudioSegment.from_file(f"{file}.mp4", format="mp4")
    # audio.export(f"{file}.wav", format="wav")
    test = TranscriptionHandler(file)
    # test.run_all_file()
    chunk_size = 50  # Process 15-second chunks
    total_duration = 170  # Example full audio length
    #
    for i in range(0, total_duration, chunk_size):
        start_time = max(0, i )
        end_time = min(start_time + chunk_size, total_duration)
        print(f"\n🔹 Processing segment {start_time}-{end_time}")
        test.run(start_time, end_time)
