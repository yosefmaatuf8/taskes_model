from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken
import requests
from odf.text import P
from io import BytesIO
from find_names import find_names
from text import split_text, save_to_odt, extract_text_from_odt
from split_recording import parse_rttm
from diarization import diarization

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=openai_api_key 
)

# Initialize tokenizer for GPT-4
tokenizer = tiktoken.encoding_for_model("gpt-4")
MAX_TOKENS = 8192  # GPT-4 limit

def process_audio_part(audio_part):
    """Transcribe a single audio part."""
    # Convert audio to MP3 format in memory
    buffer = BytesIO()
    audio_part.export(buffer, format="mp3")
    buffer.seek(0)

    # Whisper API request
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {openai_api_key}"
    }
    files = {
        "file": ("audio.mp3", buffer, "audio/mp3")
    }
    data = {
        "model": "whisper-1",
        "language": "he"
    }

    # Send request
    response = requests.post(url, headers=headers, files=files, data=data)
    if response.status_code == 200:
        return response.json()["text"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return ""
    

def transcription_with_rttm(file, max_size_mb=24.5):
    """Transcribe audio in conversational format with chronological order and combined segments."""
    # Parse speaker segments
    speaker_segments = parse_rttm(f"{file}.rttm")

    # Load audio file
    audio = AudioSegment.from_file(f"{file}.mp3", format="mp3")

    # Start processing at 0.1 seconds
    min_start_time = 0.1

    # Prepare variables
    conversation = ""
    prev_speaker = None
    combined_segment = AudioSegment.empty()

    for segment in speaker_segments:
        speaker, start_time, end_time = segment[0], segment[1], segment[2]
        if speaker == prev_speaker:
            # Extend the current combined segment
            combined_segment += audio[start_time * 1000:end_time * 1000]  # Convert seconds to milliseconds
        else:
            # Process the previous combined segment
            if combined_segment.duration_seconds > min_start_time:
                transcription_result = process_audio_part(combined_segment)
                conversation += f"{prev_speaker}: {transcription_result.strip()}\n\n"

            # Start a new segment for the new speaker
            prev_speaker = speaker
            combined_segment = audio[start_time * 1000:end_time * 1000]
            print(f"speaker: {prev_speaker}, time: {start_time}")

    # Process the last combined segment
    if combined_segment.duration_seconds > 0:
        transcription_result = process_audio_part(combined_segment)
        conversation += f"{prev_speaker}: {transcription_result.strip()}\n"
        print(f"speaker: {prev_speaker}, time: {start_time}")

    return conversation

def transcription(file, max_size_mb=24.5):
    """Split large files and transcribe each part."""
    # Check file size
    file_size = os.path.getsize(f"{file}.m4a") / (1024 * 1024)  # Convert to MB
    print(f"File size: {file_size:.2f} MB")

    # Load audio file
    audio = AudioSegment.from_file(f"{file}.m4a", format="m4a")

    # Handle large files by splitting
    if file_size > max_size_mb:
        # Calculate split parameters
        num_parts = int(file_size // max_size_mb) + 1
        print(f"File is too large. Splitting into {num_parts} smaller parts...")
        part_duration = len(audio) // num_parts
        transcription_result = ""

        # Process each part
        for i in range(num_parts):
            start_time = i * part_duration
            end_time = (i + 1) * part_duration if i < num_parts - 1 else len(audio)
            part = audio[start_time:end_time]

            # Transcribe the part
            transcription_result += process_audio_part(part) + " "
            print(f"part {i+1}/{num_parts} is transcripted")

        return transcription_result.strip()
    else:
        # Process entire file without splitting
        print("Processing entire file without splitting...")
        return process_audio_part(audio)

def summarize_chunk(chunk, names):
    """Summarize each chunk to reduce size."""
    messages = [
        {
            "role": "system",
            "content": f"Summarize the following transcription of a team meeting, preserving tasks-related names, key points and task-related information. The summarize must be written in hebrew. names is {names}"
        },
        {"role": "user", "content": chunk}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500,  # Generate a compact summary
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred during summarization: {str(e)}"


def extract_tasks(summary, names):
    """Extract tasks from the summarized transcription."""
    messages = [
        {"role": "system", "content": f"User send you a transcription of team meet and you need to return two lists, the first of tasks that have been completed, and the second of tasks that need to be completed. The tasks must be written in hebrew.\
                For each task, write who it belongs to. names is {names}"},
        {"role": "user", "content": summary}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=300,  # Output tasks only
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred during task extraction: {str(e)}"


def generate_tasks(transcription):
    """Main function to generate tasks from long transcriptions."""
    # Check token size of transcription
    names = find_names(transcription)
    token_count = len(tokenizer.encode(transcription))

    # Split transcription if needed
    if token_count > MAX_TOKENS:
        print("Transcription exceeds token limit. Splitting and summarizing...")
        chunks = split_text(transcription, MAX_TOKENS - 700)  # Leave room for prompt tokens

        # Summarize each chunk
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
            summary = summarize_chunk(chunk, names)
            print(summary)
            summaries.append(summary)

        # Combine summaries
        combined_summary = " ".join(summaries)
    else:
        combined_summary = transcription  # Use as-is if within limit

    # Extract tasks from the summarized transcription
    print("Extracting tasks from summary...")
    tasks = extract_tasks(combined_summary, names)
    return tasks

if __name__ == "__main__":
    file = "../meeting/audio1729287298"
    if not os.path.exists(f"{file}.mp3"):
        audio = AudioSegment.from_file(f"{file}.m4a", format="m4a")
        audio.export(f"{file}.mp3", format="mp3")
    if not os.path.exists(f"{file}.rttm"):
        diarization(file)
    if not os.path.exists(f"{file}.odt"):
        result_text = transcription_with_rttm(file)
        save_to_odt(result_text, f"{file}.odt")
    else:
        result_text = extract_text_from_odt(f"{file}.odt")
    print(generate_tasks(result_text))




from pydub import AudioSegment
from openai import OpenAI
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os
import tiktoken
import requests
from io import BytesIO
from odf.opendocument import OpenDocumentText, load
from odf.text import P


class AudioHandler:
    def __init__(self, huggingface_api_key=None):
        load_dotenv()
        self.api_key = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=self.api_key
        )

    def convert_to_mp3(self, file_path):
        audio = AudioSegment.from_file(file_path, format="m4a")
        mp3_path = file_path.replace(".m4a", ".mp3")
        audio.export(mp3_path, format="mp3")
        return mp3_path

    def run_diarization(self, file_path):
        diarization = self.pipeline({"uri": "audio", "audio": file_path})
        rttm_path = file_path.replace(".mp3", ".rttm")
        with open(rttm_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)
        return rttm_path

    def split_audio(self, audio, max_size_mb):
        file_size_mb = len(audio) / (1024 * 1024)
        if file_size_mb <= max_size_mb:
            return [audio]

        parts = []
        part_duration = len(audio) // (file_size_mb // max_size_mb + 1)
        for start in range(0, len(audio), part_duration):
            parts.append(audio[start:start + part_duration])
        return parts


class TaskManager:
    @staticmethod
    def save_to_odt(text, filename):
        doc = OpenDocumentText()
        for line in text.split("\n"):
            if line.strip():
                doc.text.addElement(P(text=line))
                doc.text.addElement(P(text=""))
        doc.save(filename)

    @staticmethod
    def extract_text_from_odt(filename):
        doc = load(filename)
        return "\n".join(str(p) for p in doc.getElementsByType(P))

class TranscriptionHandler:
    def __init__(self, openai_api_key, huggingface_api_key=None, language="he"):
        self.api_key = openai_api_key
        self.language = language
        self.client = OpenAI(api_key=self.api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = 8192

        self.audio_handler = AudioHandler(huggingface_api_key)
        self.task_manager = TaskManager()


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
                print(names_string)

                names = self.parse_names(names_string, known_names)
                known_names.update(names)
        else:
            names_string = self.find_in_chunk(transcription, names_string)
            names = self.parse_names(names_string, known_names)
            known_names.update(names)

        print(f"Final speaker names: {known_names}")
        return known_names

    def split_text(self, text, max_tokens):
        """Split text into chunks within the token limit."""
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(self.tokenizer.encode(" ".join(current_chunk + [word]))) > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    # Other existing methods remain as before...
    # def transcribe_audio(self, audio_segment):
    #     buffer = BytesIO()
    #     audio_segment.export(buffer, format="mp3")
    #     buffer.seek(0)
    #
    #     url = "https://api.openai.com/v1/audio/transcriptions"
    #     headers = {"Authorization": f"Bearer {self.api_key}"}
    #     files = {"file": ("audio.mp3", buffer, "audio/mp3")}
    #     data = {"model": "whisper-1", "language": self.language}
    #
    #     response = requests.post(url, headers=headers, files=files, data=data)
    #     if response.status_code == 200:
    #         return response.json()["text"]
    #     else:
    #         raise Exception(f"Error in transcription: {response.status_code}, {response.text}")
    #
    # def transcription_with_rttm(self, file_path, max_size_mb=24.5):
    #     # Convert audio if necessary
    #     mp3_path = file_path if file_path.endswith(".mp3") else self.audio_handler.convert_to_mp3(file_path)
    #     rttm_path = mp3_path.replace(".mp3", ".rttm")
    #     if not os.path.exists(rttm_path):
    #         self.audio_handler.run_diarization(mp3_path)
    #
    #     # Parse diarization segments
    #     with open(rttm_path, "r") as rttm_file:
    #         speaker_segments = [
    #             line.strip().split() for line in rttm_file.readlines() if not line.startswith("#")
    #         ]
    #
    #     audio = AudioSegment.from_file(mp3_path, format="mp3")
    #     conversation = ""
    #     prev_speaker = None
    #     combined_segment = AudioSegment.empty()
    #
    #     for segment in speaker_segments:
    #         speaker, start_time, end_time = segment[0], float(segment[3]), float(segment[4])
    #         if speaker == prev_speaker:
    #             combined_segment += audio[start_time * 1000:end_time * 1000]
    #         else:
    #             if combined_segment.duration_seconds > 0:
    #                 transcription_result = self.transcribe_audio(combined_segment)
    #                 conversation += f"{prev_speaker}: {transcription_result.strip()}\n\n"
    #
    #             prev_speaker = speaker
    #             combined_segment = audio[start_time * 1000:end_time * 1000]
    #
    #     # Process the last segment
    #     if combined_segment.duration_seconds > 0:
    #         transcription_result = self.transcribe_audio(combined_segment)
    #         conversation += f"{prev_speaker}: {transcription_result.strip()}\n"
    #
    #     return conversation

    # def process_transcription(self, file_path, max_size_mb=24.5):
    #     audio = AudioSegment.from_file(file_path, format="mp3")
    #     audio_parts = self.audio_handler.split_audio(audio, max_size_mb)
    #     return " ".join(self.transcribe_audio(part) for part in audio_parts)

    def generate_tasks(self, transcription):
        # Find names and process tasks
        def split_text(text, max_tokens):
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

        token_count = len(self.tokenizer.encode(transcription))
        if token_count > self.max_tokens:
            chunks = split_text(transcription, self.max_tokens - 700)
            summaries = [self.summarize_transcription(chunk) for chunk in chunks]
            transcription = " ".join(summaries)

        return self.extract_tasks(transcription)

    def summarize_transcription(self, transcription):
        messages = [{"role": "user", "content": f"Summarize this meeting transcript: {transcription}"}]
        response = self.client.chat.completions.create(model="gpt-4", messages=messages, max_tokens=500)
        return response.choices[0].message.content

    def extract_tasks(self, summarized_text):
        messages = [
            {"role": "user", "content": f"Extract tasks from this text: {summarized_text}"}
        ]
        response = self.client.chat.completions.create(model="gpt-4", messages=messages, max_tokens=300)
        return response.choices[0].message.content
