from pyannote.audio import Model
from pyannote.audio import Inference
from scipy.spatial.distance import cosine
import numpy as np
import os
from pydub import AudioSegment
from io import BytesIO
from dotenv import load_dotenv

def load_model(api_key):
    # Initialize the model
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=api_key)
    inference = Inference(model, window="whole")
    return inference

# Function to split audio into segments
def split_audio(audio_path, segment_lengths):
    audio = AudioSegment.from_file(f"{audio_path}.wav")
    audio_segments = []

    for length in segment_lengths:
        # Convert segment length from seconds to milliseconds
        segment_duration_ms = int(length * 1000)

        # If the segment is longer than the audio, take the full audio
        if segment_duration_ms > len(audio):
            audio_segments.append(audio)
        else:
            for start_time in range(0, len(audio), segment_duration_ms):
                end_time = start_time + segment_duration_ms
                if end_time <= len(audio):
                    audio_segments.append(audio[start_time:end_time])
    
    return audio_segments

# Function to compare embeddings and find the closest speaker
def get_closest_speaker(segment_embedding, track_embeddings):
    closest_speaker = None
    min_distance = float("inf")
    
    for speaker, track_embedding in track_embeddings.items():
        # Compute cosine similarity between segment embedding and track embedding
        distance = cosine(segment_embedding, track_embedding)
        
        if distance < min_distance:
            min_distance = distance
            closest_speaker = speaker
    
    return closest_speaker, min_distance

def assign_speakers_to_clips(audio_clips, embeddings, inference, tresh_hold):
    speakers = []
    speaker = 0

    for clip in audio_clips:
        buffer = BytesIO()
        if len(clip["clip"]) < 400:
            speakers.append("None")
            continue
        clip["clip"].export(buffer, format="wav")
        buffer.seek(0)
        embedding = inference(buffer)

        if not embeddings:
            embeddings[f"speaker_{speaker}"] = embedding
        else:
            segment_embedding = embedding
            closest_speaker, min_distance = get_closest_speaker(segment_embedding, embeddings)
            if min_distance > tresh_hold:
                speaker = len(embeddings)
                embeddings[f"speaker_{speaker}"] = segment_embedding
            else:
                speaker = int(closest_speaker.split('_')[1])
        speakers.append(f"speaker_{speaker}")
    return speakers, embeddings

if __name__ == "__main__":
    # Define your parameters
    load_dotenv()
    audio_file_path = "../meeting/audio1725163871"
    api_key = os.getenv("OPENAI_API_KEY")
    SEGMENT_LENGTHS = [3]  # Length of each segment in seconds
    tresh_hold = 0.8

    # Load the model
    inference = load_model(api_key)

    # Split the audio into segments
    audio_clips = split_audio(audio_file_path, SEGMENT_LENGTHS)

    # Wrap audio segments into dictionary for processing
    audio_clips_dicts = [{"clip": clip} for clip in audio_clips]

    # Initialize an empty dictionary for embeddings
    embeddings = {}

    # Assign speakers to each audio clip
    speakers, updated_embeddings = assign_speakers_to_clips(audio_clips_dicts, embeddings, inference, tresh_hold)

    # Print the results
    for i, speaker in enumerate(speakers):
        print(f"Segment {i + 1}: {speaker}")