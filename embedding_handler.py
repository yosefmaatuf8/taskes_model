from scipy.spatial.distance import cosine

import numpy as np
from io import BytesIO
from pyannote.audio import Model, Inference
from globals import GLOBALS
from dotenv import load_dotenv

class EmbeddingHandler:
    """
    Handles speaker diarization using speaker embeddings.

    This class provides a comprehensive pipeline for speaker diarization, which involves identifying "who spoke when" in an audio recording.
    It leverages pre-trained speaker embedding models from Hugging Face, specifically the 'pyannote/embedding' model, to generate numerical
    representations (embeddings) of speaker voices. These embeddings are then used to compare and cluster audio segments based on speaker
    similarity.

    **Process Overview:**

    1.  **Model Loading:** The class initializes by loading a pre-trained speaker embedding model using the Hugging Face API.
    2.  **Speaker Assignment:**
         * If the class has no previously known speakers, it uses the similarity score to determine if the speaker is consistent. If so, a new speaker is added. Otherwise, the speaker is marked as unknown.
         * If the class has known speakers, it compares the embedding of the current audio segment to the embeddings of known speakers. If a close match is found (below a defined distance threshold), the known speaker is assigned.
         * If no close match is found, the similarity score is used. If high enough, a new speaker is added. Otherwise, the speaker is marked as unknown.
    3.  **Similarity Computation:** For audio segments, the class computes the similarity of the speaker within those segments. This is done by splitting the audio into smaller chunks, generating embeddings for each chunk, and comparing them. High similarity scores indicate a consistent speaker within the segment. 
    4.  **Embedding Extraction:** Given an audio clip, the class extracts a speaker embedding, a numerical representation of the speaker's voice.
    5.  **Transcription Processing:** The class can process transcriptions, assigning speaker labels to each segment based on the embedding analysis.
    6.  **Database Integration:** Optionally, the class can integrate with a database to store and retrieve speaker embeddings, allowing for persistent speaker identification across multiple audio recordings.
    7.  **Unknown Speaker Handling:** The class maintains a separate record of unknown speakers, which can be reclassified later using a callback function.

    **Key Features:**

    * Uses pre-trained speaker embedding models for high accuracy.
    * Handles both known and unknown speakers.
    * Computes speaker similarity within audio segments.
    * Integrates with databases for persistent speaker data.
    * Provides a callback mechanism for reclassifying unknown speakers.
    * Processes transcriptions and assigns speaker labels.

    **Usage:**

    This class is intended to be used in applications that require speaker diarization, such as meeting transcription, call center analysis, and multimedia content analysis.
    """
    def __init__(self, db_manager=None, reclassify_callback=None):
        """
        Initializes the EmbeddingHandler.

        Args:
            db_manager (optional): An object for managing speaker embeddings in a database. Defaults to None.
            reclassify_callback (optional): A callback function to reclassify unknown speakers. Defaults to None.
        """
        load_dotenv()
        self.db_manager = db_manager
        self.huggingface_api_key = GLOBALS.huggingface_api_key
        self.inference = self.load_model()
        self.embeddings = db_manager.load_user_embeddings() if db_manager else {}
        self.unknown_embeddings = {}
        self.threshold_closest = 0.8
        self.threshold_similarity = 0.1
        self.segment_length = 2 * 1000
        self.num_segments = 2
        self.speakers = []
        self.speaker_count = 0
        self.reclassify_callback = reclassify_callback  # Callback to reclassify unknown speakers

    def load_model(self):
        """Loads the pre-trained speaker embedding model."""
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=self.huggingface_api_key)
        return Inference(model, window="whole")
    
    def extract_embedding(self, audio_clip):
        """
        Extracts the embedding from an audio clip.

        Args:
            audio_clip (pydub.AudioSegment): The audio clip to extract the embedding from.

        Returns:
            numpy.ndarray: The speaker embedding.
        """
        buffer = BytesIO()
        audio_clip.export(buffer, format="wav")
        buffer.seek(0)
        return self.inference(buffer)

    def compute_similarity(self, audio):
        """
        Splits an audio signal into segments, computes embeddings for each segment, 
        and calculates the similarity scores between all pairs of segments.

        The audio is divided into a number of equal-length segments, and for each segment, 
        an embedding is extracted. Then, the cosine similarity between each pair of segment 
        embeddings is calculated. The function returns the mean similarity score or None 
        if the audio signal is too short or there is an error during processing.

        :param audio: A numpy array representing the audio signal.
        :return: A float representing the mean similarity score between segments, 
                or None if there is insufficient data or an error occurs.
        """
        if len(audio) < self.num_segments * self.num_segments:
            return None

        embeddings = []

        try:
            for i in range(self.num_segments):
                start = i * self.segment_length
                end = (i + 1) * self.segment_length
                segment = audio[start:end]
                embedding = self.extract_embedding(segment)
                embeddings.append(embedding)

            similarities = [
                1 - cosine(embeddings[i], embeddings[j])
                for i in range(self.num_segments) for j in range(i + 1, self.num_segments)
            ]

            return np.mean(similarities) if similarities else None

        except Exception as e:
            print(f"Error in compute_similarity: {e}")
            return None


    def get_closest_speaker(self,segment_embedding, track_embeddings=None):
        """
        Finds the closest speaker to a given embedding.

        Args:
            segment_embedding (numpy.ndarray): The embedding to compare.
            track_embeddings (dict, optional): A dictionary of known speaker embeddings. Defaults to self.embeddings.

        Returns:
            tuple: The closest speaker label and the minimum distance.
        """
        if track_embeddings is None:
            track_embeddings = self.embeddings
        closest_speaker = None
        min_distance = float("inf")

        for speaker, track_embedding in track_embeddings.items():
            # Calculate the cosine distance between embeddings
            distance = cosine(segment_embedding, track_embedding)

            if distance < min_distance:
                min_distance = distance
                closest_speaker = speaker

        return closest_speaker, min_distance
    
    def handle_new_speaker(self, audio_clip):
        """
        Handles the logic for a new speaker.

        Args:
            audio_clip (pydub.AudioSegment): The audio clip of the new speaker.

        Returns:
            str: The label of the new speaker.
        """
        buffer = BytesIO()
        audio_clip[:self.segment_length * self.num_segments].export(buffer, format="wav")
        embedding = self.extract_embedding(audio_clip)
        speaker_label = f"speaker_{self.speaker_count}"
        self.embeddings[speaker_label] = embedding
        print(f"add {speaker_label}")
        self.speakers.append(speaker_label)
        self.speaker_count += 1
        if self.db_manager:
            self.db_manager.save_user_embeddings(self.embeddings)
        if self.reclassify_callback:
            self.reclassify_callback()
        return speaker_label
    
    def handle_unknown_speaker(self, embedding):
        """
        Handles the logic for an unknown speaker.

        Args:
            embedding (numpy.ndarray): The embedding of the unknown speaker.

        Returns:
            str: The label of the unknown speaker.
        """
        speaker_label = f"unknown_{len(self.unknown_embeddings)}"
        self.unknown_embeddings[speaker_label] = embedding
        return speaker_label

    def assign_speaker_with_similarity(self, audio_clip):
        """
        Assigns a speaker to an audio clip using similarity checks.

        Args:
            audio_clip (pydub.AudioSegment): The audio clip to assign a speaker to.

        Returns:
            str: The assigned speaker label.
        """
        if len(audio_clip) < 400:
            return "Little_segment"

        embedding = self.extract_embedding(audio_clip)

        if not self.embeddings:
            similarity_score = self.compute_similarity(audio_clip)
            if similarity_score and similarity_score > self.threshold_similarity:
                return self.handle_new_speaker(audio_clip)
            return self.handle_unknown_speaker(embedding)

        closest_speaker, min_distance = self.get_closest_speaker(embedding, self.embeddings)
        
        if min_distance <= self.threshold_closest:
            return closest_speaker
        
        similarity_score = self.compute_similarity(audio_clip)

        if similarity_score and similarity_score > self.threshold_similarity:
            return self.handle_new_speaker(audio_clip)
        
        return self.handle_unknown_speaker(embedding)

    def process_transcription_with_clustering(self, json_transcription, audio_chunk):
        """
        Processes a transcription with speaker clustering.

        Args:
            json_transcription (dict): The JSON transcription.
            audio_chunk (pydub.AudioSegment): The audio chunk.

        Returns:
            list: A list of dictionaries representing the conversation with speaker labels.
        """
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