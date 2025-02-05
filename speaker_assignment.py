from pyannote.audio import Model, Inference
from scipy.spatial.distance import cosine
import numpy as np
import os
from pydub import AudioSegment
from io import BytesIO
from dotenv import load_dotenv
from score_record import compute_similarity

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

def embedding(clip, inference):
    buffer = BytesIO()
    clip.export(buffer, format="wav")
    buffer.seek(0)
    return inference(buffer)

def assign_speakers_to_clips(audio_clips, embeddings, inference, treshold_closest, trehold_similarity, unknow):
    speakers = []
    speaker = 0

    for clip in audio_clips:
        if len(clip["clip"]) < 400:
            continue

        similarity = compute_similarity(clip["clip"], 1000, inference)
        if not embeddings and similarity < trehold_similarity:
            unknow.append(clip)
            continue

        segment_embedding = embedding(clip["clip"], inference)

        if not embeddings:
            embeddings[f"speaker_{speaker}"] = embedding(clip["clip"][:8000], inference)
            speakers.append({
                "start": clip["start"],
                "end": clip["end"],
                "clip": clip["clip"],
                "speaker": f"speaker_{speaker}",
                "distance": 0,
                "idx": clip["idx"]
            })
            continue
        closest_speaker, min_distance = get_closest_speaker(segment_embedding, embeddings)
        if similarity < trehold_similarity and min_distance > treshold_closest:
            unknow.append({
                "start": clip["start"],
                "end": clip["end"],
                "clip": clip["clip"],
                "speaker": f"speaker_{speaker}",
                "distance": min_distance,
                "idx": clip["idx"]                
            })
        else:
            if min_distance > treshold_closest:
                speaker = len(embeddings)
                embeddings[f"speaker_{speaker}"] = embedding(clip["clip"][:8000], inference)
            else:
                speaker = int(closest_speaker.split('_')[1])
            speakers.append({
                "start": clip["start"],
                "end": clip["end"],
                "clip": clip["clip"],
                "speaker": f"speaker_{speaker}",
                "distance": min_distance,
                "idx": clip["idx"]
            })
    return speakers, embeddings, unknow

if __name__ == "__main__":
    # Define your parameters
    load_dotenv()
    audio_file_path = "../meeting/audio1725163871"
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    SEGMENT_LENGTHS = [3]  # Length of each segment in seconds
    tresh_hold = 1.0

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

    # Create a directory to save the final speaker audio files
    if not os.path.exists(audio_file_path):
        os.makedirs(audio_file_path)

    # Create a dictionary to store combined audio per speaker
    speaker_audio = {}

    for idx, speaker in enumerate(speakers):
        # Initialize or append the audio for the speaker
        if speaker not in speaker_audio:
            speaker_audio[speaker] = AudioSegment.empty()

        speaker_audio[speaker] += audio_clips_dicts[idx]["clip"]

    # Save each speaker's combined audio to a single file
    for speaker, combined_audio in speaker_audio.items():
        # Save the combined audio for each speaker
        combined_audio.export(os.path.join(audio_file_path, f"{speaker}_combined.wav"), format="wav")

    # Print the results
    for i, speaker in enumerate(speakers):
        print(f"Segment {i + 1}: {speaker}")
    print(f"Number of speakers identified: {len(embeddings)}")