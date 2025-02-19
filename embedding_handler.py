from scipy.spatial.distance import cosine

import numpy as np
from io import BytesIO
from pyannote.audio import Model, Inference
from globals import GLOBALS
from dotenv import load_dotenv

class EmbeddingHandler:
    def __init__(self, db_manager=None, reclassify_callback=None):
        load_dotenv()
        self.db_manager = db_manager
        self.huggingface_api_key = GLOBALS.huggingface_api_key
        self.inference = self.load_model()
        self.embeddings = db_manager.load_user_embeddings() if db_manager else {}
        self.unknown_embeddings = {}
        self.threshold_closest = 0.7
        self.threshold_similarity = 0.2
        self.speakers = []
        self.speaker_count = 0
        self.reclassify_callback = reclassify_callback  # Callback to reclassify unknown speakers

    def load_model(self):
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=self.huggingface_api_key)
        return Inference(model, window="whole")


    def compute_similarity(self, audio):
        """
        Splits a 10-second audio into 2-second chunks, computes embeddings, and returns similarity scores.
        :param audio: Audio signal (numpy array).
        :return: Mean similarity score between segments (float) or None if insufficient data.
        """

        # Ensure there is enough data
        if len(audio) < 3000:
            return None

        embeddings = []

        try:

            segment_length = len(audio) / 3  # 2 seconds in samples
            num_segments = 3  # 6 seconds split into 3 segments of 2 seconds

            for i in range(num_segments):
                start = i * segment_length
                end = (i + 1) * segment_length
                segment = audio[start:end]

                # Convert to a file-like object in memory (BytesIO)
                buffer = BytesIO()
                segment.export(buffer, format="wav")
                buffer.seek(0)

                embedding = self.inference(buffer)
                embeddings.append(embedding)

                # Compute similarity between all possible pairs
            similarities = [
                1 - cosine(embeddings[i], embeddings[j])
                for i in range(num_segments) for j in range(i + 1, num_segments)
            ]

            return np.mean(similarities) if similarities else None

        except Exception as e:
            print(f"Error in compute_similarity: {e}")
            return None


    def get_closest_speaker(self,segment_embedding, track_embeddings=None):
        if track_embeddings is None:
            track_embeddings = self.embeddings
        """Compare embeddings and find the closest speaker."""
        closest_speaker = None
        min_distance = float("inf")

        for speaker, track_embedding in track_embeddings.items():
            # Calculate the cosine distance between embeddings
            distance = cosine(segment_embedding, track_embedding)

            if distance < min_distance:
                min_distance = distance
                closest_speaker = speaker

        return closest_speaker, min_distance

    def assign_speaker_with_similarity(self, audio_clip):
        """Assigns a speaker using similarity checks."""
        buffer = BytesIO()
        audio_clip.export(buffer, format="wav")
        if len(audio_clip) < 400:
            return "Little_segment"
        buffer.seek(0)
        embedding = self.inference(buffer)

        if not self.embeddings:
            similarity_score = self.compute_similarity(audio_clip)
            if similarity_score and similarity_score > self.threshold_similarity:
                speaker_label = f"speaker_{self.speaker_count}"
                audio_clip[:6000].export(buffer, format="wav")
                buffer.seek(0)
                embedding = self.inference(buffer)
                self.embeddings[speaker_label] = embedding
                print(f"add {speaker_label}")
                self.speakers.append(speaker_label)
                self.speaker_count += 1
                if self.reclassify_callback:
                    self.reclassify_callback()
            else:
                speaker_label = f"unknown_{len(self.unknown_embeddings)}"
                self.unknown_embeddings[speaker_label] = embedding
            return speaker_label
        closest_speaker, min_distance = self.get_closest_speaker(embedding, self.embeddings)

        if min_distance < self.threshold_closest:
            speaker_label = closest_speaker
            return speaker_label  # Return early if a close match is foundelif not similarity_score or similarity_score < self.threshold_similarity:
        # Only calculate similarity if no close match was found
        similarity_score = self.compute_similarity(audio_clip)

        if similarity_score and similarity_score > self.threshold_similarity:
            speaker_label = f"speaker_{self.speaker_count}"
            audio_clip[:6000].export(buffer, format="wav")
            buffer.seek(0)
            embedding = self.inference(buffer)
            self.embeddings[speaker_label] = embedding
            print(f"add {speaker_label}")
            self.speakers.append(speaker_label)
            self.speaker_count += 1
            if self.db_manager:
                self.db_manager.save_user_embeddings(self.embeddings)
            if self.reclassify_callback:
                self.reclassify_callback()

        else:
            speaker_label = f"unknown_{len(self.unknown_embeddings)}"
            self.unknown_embeddings[speaker_label] = embedding
        return speaker_label

    def process_transcription_with_clustering(self, json_transcription, audio_chunk):
        speakers = []
        segments = json_transcription["segments"]
        for segment in segments:
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            segment_audio = audio_chunk[start_ms:end_ms]
            speaker_label = self.assign_speaker_with_similarity(segment_audio)
            speakers.append(speaker_label)
        conversation = [{"start": seg["start"], "end": seg["end"], "text": seg["text"], "speaker": speakers[i]} for i, seg in enumerate(segments)]
        return conversation