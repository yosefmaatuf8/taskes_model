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
