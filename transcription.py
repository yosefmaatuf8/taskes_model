from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken
import requests
from io import BytesIO
from text import save_to_odt
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

if __name__ == "__main__":
    file = "../meeting/audio1729287298"
    if not os.path.exists(f"{file}.mp3"):
        audio = AudioSegment.from_file(f"{file}.m4a", format="m4a")
        audio.export(f"{file}.mp3", format="mp3")
    if not os.path.exists(f"{file}.rttm"):
        diarization(file)
    result_text = transcription_with_rttm(file)
    save_to_odt(result_text, f"{file}.odt")
