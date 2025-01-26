from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken
import requests
from io import BytesIO
from globals import GLOBALS

class TranscriptionHandler:
    def __init__(self, language=GLOBALS.LANGUAGE, mp3_path=None, rttm_path=None, speaker_segments=None):
        self.api_key = GLOBALS.API_KEY
        self.language = language
        self.client = OpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = GLOBALS.max_tokens
        self.mp3_path = mp3_path
        self.rttm_path = rttm_path
        self.speaker_segments = speaker_segments
        self.max_audio_duration = 600  # מקסימום משך קובץ שמע (בשניות) לשליחה ל-Whisper

    def transcribe_audio_with_timestamps(self, audio_segment):
        """Transcribe a single audio segment using OpenAI Whisper with timestamps."""
        buffer = BytesIO()
        audio_segment.export(buffer, format="mp3")
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
            print(f"Audio duration is {audio_duration}s, exceeding the limit of {self.max_audio_duration}s. Splitting...")
            chunks = []
            for start_time in range(0, int(audio_duration), self.max_audio_duration):
                end_time = min(start_time + self.max_audio_duration, int(audio_duration))
                chunk = audio[start_time * 1000:end_time * 1000]
                chunks.append(chunk)
            return chunks
        else:
            return [audio]

    def process_transcription_with_speakers(self):
        """Process audio, assign transcription to speakers, and return the full transcription."""
        audio = AudioSegment.from_file(self.mp3_path, format="mp3")
        audio_chunks = self.split_audio_if_needed(audio)
        conversation = []

        for i, chunk in enumerate(audio_chunks):
            print(f"Processing audio chunk {i + 1}/{len(audio_chunks)}...")
            transcription_data = self.transcribe_audio_with_timestamps(chunk)

            # Parse RTTM and speaker information
            for segment in transcription_data["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]

                # Match segment to speaker
                speaker = "unknown"
                for seg_speaker, seg_start, seg_end in self.speaker_segments:
                    if start_time >= seg_start and end_time <= seg_end:
                        speaker = seg_speaker
                        break

                # Add to conversation
                conversation.append(f"{speaker}: {text.strip()}")

        # Combine into a single transcription
        transcription = "\n".join(conversation)

        # Check if transcription exceeds token limit
        if len(self.tokenizer.encode(transcription)) > self.max_tokens:
            print("Transcription exceeds token limit. Splitting into chunks...")
            return self.split_and_process_transcription(conversation)

        return transcription


    def split_and_process_transcription(self, conversation):
        """Split long transcription into chunks and process each separately."""
        chunks = []
        current_chunk = []
        current_tokens = 0

        for line in conversation:
            tokens = len(self.tokenizer.encode(line))
            if current_tokens + tokens > self.max_tokens - 500:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_tokens = tokens
            else:
                current_chunk.append(line)
                current_tokens += tokens

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def infer_speaker_names(self, transcription):
        """Send transcription to LLM to infer speaker names."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You will receive a transcript of a conversation where speakers are "
                    "labeled as 'Speaker 0', 'Speaker 1', etc. Your task is to infer the actual "
                    "names of the speakers based on their speech content. The possible names are: "
                    "אברהם, אמיר, שוקי, בנצי, יוסף, יהודה, דוד ג., דוד ש. "
                    "If you cannot determine the name of a speaker, label them as 'unknown'."
                ),
            },
            {"role": "user", "content": transcription}
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
            return f"An error occurred during speaker name inference: {str(e)}"

    def process_with_speaker_inference(self):
        """Main method to handle transcription and speaker inference."""
        transcription = self.process_transcription_with_speakers()

        if isinstance(transcription, list):  # If transcription was split into chunks
            full_transcription = ""
            for i, chunk in enumerate(transcription):
                print(f"Inferring speaker names for chunk {i + 1}/{len(transcription)}...")
                inferred_names = self.infer_speaker_names(chunk)
                full_transcription += inferred_names + "\n"
            return full_transcription.strip()

        # If transcription is a single chunk
        print("Inferring speaker names...")
        return self.infer_speaker_names(transcription)

# Usage example:
# handler = TranscriptionHandler(
#     mp3_path="path/to/your/file.mp3",
#     speaker_segments=[
#         ("Speaker 0", 0, 10),
#         ("Speaker 1", 10, 20),
#         ("Speaker 0", 20, 30),
#     ]
# )
# print(handler.process_with_speaker_inference())
