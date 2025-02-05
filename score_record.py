import numpy as np
import torch
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cosine
from pydub import AudioSegment
from io import BytesIO

def load_model(api_key):
    # Initialize the speaker embedding model
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=api_key)
    inference = Inference(model, window="whole")
    return inference

def compute_similarity(audio, sr, inference):
    """
    Splits a 10-second audio into 2-second chunks, computes embeddings, and returns similarity scores.
    
    :param audio: Audio signal (numpy array).
    :param sr: Sampling rate of the audio.
    :param inference: Preloaded PyAnnote embedding model.
    :return: List of similarity scores between consecutive 2-second segments.
    """
    segment_length = 2 * sr  # 2 seconds in samples
    num_segments = 3  # 10 sec / 2 sec
    
    # Ensure audio is exactly 10 seconds
    if len(audio) < 6 * sr:
        return 0
    
    embeddings = []
    buffer = BytesIO()
    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segment = audio[start:end]
        segment.export(buffer, format="wav")
        buffer.seek(0)
        embedding = inference(buffer)
        embeddings.append(embedding)
    
    # Compute all pairwise similarities
    similarities = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            similarity = 1 - cosine(embeddings[i], embeddings[j])
            similarities.append(similarity)
    
    return np.mean(similarities)

# Example usage
if __name__ == "__main__":
    api_key = "your_huggingface_api_key"
    inference = load_model(api_key)
    
    similarities = []
    for idx in range(616):
        # Load a 10-second .wav file (replace 'audio.wav' with your file path)
        audio = AudioSegment.from_wav(f"../meeting/audio1725163871/all/{idx}")
        similarities.append((compute_similarity(audio, 1000, inference), idx))

    similarities.sort()
    print(similarities)
        
