from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken
import requests
from odf.text import P
from io import BytesIO
from globals import GLOBALS

class TranscriptionHandler:
    def __init__(self, language=GLOBALS.LANGUAGE, mp3_path=None, rttm_path=None, speaker_segments=None):
        self.api_key = GLOBALS.API_KEY
        self.language = language
        self.client = OpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = 8192
        self.mp3_path = mp3_path
        self.rttm_path = rttm_path
        self.speaker_segments = speaker_segments

    def transcribe_audio(self, audio_segment):
        """Transcribe a single audio segment using OpenAI Whisper."""
        buffer = BytesIO()
        audio_segment.export(buffer, format="mp3")
        buffer.seek(0)

        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"file": ("audio.mp3", buffer, "audio/mp3")}
        data = {"model": "whisper-1", "language": self.language}

        response = requests.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            return response.json()["text"]
        else:
            raise Exception(f"Error in transcription: {response.status_code}, {response.text}")

    def transcription_with_rttm(self):
        """Process audio and assign transcription to speakers based on RTTM."""
        audio = AudioSegment.from_file(self.mp3_path, format="mp3")
        conversation = ""
        prev_speaker = None
        combined_segment = AudioSegment.empty()

        for speaker, start_time, end_time in self.speaker_segments:
            if speaker == prev_speaker:
                combined_segment += audio[start_time * 1000:end_time * 1000]
            else:
                if combined_segment.duration_seconds > 0:
                    transcription_result = self.transcribe_audio(combined_segment)
                    conversation += f"{prev_speaker}: {transcription_result.strip()}\n\n"

                prev_speaker = speaker
                combined_segment = audio[start_time * 1000:end_time * 1000]

        if combined_segment.duration_seconds > 0:
            transcription_result = self.transcribe_audio(combined_segment)
            conversation += f"{prev_speaker}: {transcription_result.strip()}\n"

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
    def find_in_chunk(self, chunk, names_string):
        """Infer speaker names from a chunk of transcription."""
        messages = [
            {
                "role": "system",
                "content": (
                    "The user will have a transcript of a meeting with user separation. "
                    "You are required to return who is speaker 0, etc. "
                    "Avraham is manager of the meet, the members are: Avraham, Amir, Shuky, "
                    "Benzy, Yosef, Yehuda, David G. and David S. The order and numbering "
                    "of the speakers says nothing about their names, you have to infer names "
                    "only from the transcription. If there is a speaker that you don't know, "
                    "write 'unknown'."
                ),
            },
            {
                "role": "system",
                "content": f"These names are evaluated based on the previous parts of the conversation: {names_string}. Keep them unless you have proof to the contrary.",
            },
            {"role": "user", "content": chunk},
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=300,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred during name inference: {str(e)}"

    def parse_names(self, names_string, known_names):
        """Parse the names string into a dictionary of speaker labels."""
        names_dict = {}
        lines = names_string.split("\n")
        for line in lines:
            if line.strip():  # Skip empty lines
                speaker, name = line.split(":", 1)
                speaker = speaker.strip()
                name = name.strip()

                # Only add 'unknown' if the speaker is not in known_names
                if name.lower() == 'unknown' and speaker not in known_names:
                    names_dict[speaker] = name
                elif name.lower() != 'unknown':
                    names_dict[speaker] = name
        return names_dict

    def find_names(self, transcription):
        """Find names from a long transcription."""
        token_count = len(self.tokenizer.encode(transcription))
        known_names = {}
        names_string = ""

        if token_count > self.max_tokens:
            print("Transcription exceeds token limit. Splitting and processing...")
            chunks = self.split_text(transcription, self.max_tokens - 700)

            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i + 1}/{len(chunks)}...")
                names_string = self.find_in_chunk(chunk, names_string)
                names = self.parse_names(names_string, known_names)
                known_names.update(names)
        else:
            names_string = self.find_in_chunk(transcription, names_string)
            names = self.parse_names(names_string, known_names)
            known_names.update(names)

        return known_names

    def transcription_with_speakers(self):
        """Transcribe audio, assign speaker names, and return the full transcription."""
        conversation = self.transcription_with_rttm()

        print("Inferring speaker names...")
        names = self.find_names(conversation)
        for speaker_id, name in names.items():
            conversation = conversation.replace(f"{speaker_id}:", f"{name}:")

        return conversation
