import re
import json
from pydub import AudioSegment
import librosa
import numpy as np
from tqdm import tqdm

from globals import GLOBALS
def split_text(tokenizer, text, max_tokens=GLOBALS.max_tokens):
    """Splits text into chunks ensuring it does not exceed max tokens per chunk."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        new_chunk = " ".join(current_chunk + [word])
        if len(tokenizer.encode(new_chunk)) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def extract_json_from_code_block(response_text):
    """
    Extracts JSON from a response text and attempts to fix common issues
    such as unbalanced brackets and missing closures.
    Returns a valid JSON object or None if it cannot be fixed.
    """

    # Extract JSON content from triple backticks
    pattern = r'```json(.*?)```'
    match = re.search(pattern, response_text, flags=re.DOTALL | re.IGNORECASE)
    json_str = match.group(1).strip() if match else response_text.strip()

    # Function to fix unbalanced brackets
    def fix_json_string(json_str):
        stack = []
        fixed_str = ""
        for char in json_str:
            if char in "{[":
                stack.append(char)
            elif char in "}]":
                if stack and ((char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '[')):
                    stack.pop()
                else:
                    continue  # Ignore excess closing brackets
            fixed_str += char

        while stack:
            open_bracket = stack.pop()
            fixed_str += '}' if open_bracket == '{' else ']'

        return fixed_str

    # Function to clean newlines inside JSON strings
    def clean_json_string(json_str):
        return re.sub(r"\n\s*", " ", json_str)  # Replace newlines with spaces within the JSON string

    # First attempt: clean newlines and parse as-is
    cleaned_json_str = clean_json_string(json_str)
    try:
        return json.loads(cleaned_json_str)
    except json.JSONDecodeError:
        pass

    # Second attempt: fix unbalanced brackets
    fixed_json_str = fix_json_string(cleaned_json_str)
    try:
        return json.loads(fixed_json_str)
    except json.JSONDecodeError:
        pass

    # Third attempt: append closing brackets if missing
    if not json_str.endswith('}]'):
        json_str_fixed = json_str + "}]"
        try:
            return json.loads(json_str_fixed)
        except json.JSONDecodeError:
            pass

    print("Failed to parse JSON.: ", response_text)
    return None

def remove_silent_parts(audio_data, threshold, margin): # Takes audio data and sample rate
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



def load_and_clean_audio(audio_file, format, chunk_duration=800, threshold=-10, margin=500):  # Add chunk_duration
    """Loads, cleans audio in chunks, and returns a combined AudioSegment."""
    try:
        audio_segment = AudioSegment.from_file(audio_file)
        sr = audio_segment.frame_rate
        total_duration = len(audio_segment)  # in milliseconds
        cleaned_audio_chunks = []

        for start in tqdm(range(0, total_duration, chunk_duration * 1000), desc="Processing audio chunks"):
            end = min(start + chunk_duration * 1000, total_duration)
            chunk = audio_segment[start:end]

            samples = np.array(chunk.get_array_of_samples(), dtype=np.int16)
            cleaned_samples = remove_silent_parts(samples, threshold, margin)

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
            cleaned_audio_segment.export(f"{audio_file[:-4]}_clean.wav", format=format)
            return cleaned_audio_segment
        else:
            return None

    except Exception as e:
        print(f"Error loading or cleaning audio: {e}")
        return None
    