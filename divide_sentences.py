from pydub import AudioSegment
import requests
import os
from dotenv import load_dotenv
from speaker_assignment import load_model, assign_speakers_to_clips
from tqdm import tqdm
import json
from split_recording import parse_rttm, split_audio, split_by_clip

def transcribe_audio_with_whisper(audio_chunk, api_key, language='he'):
    """Transcribe audio chunk using OpenAI Whisper API."""
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": audio_chunk}
    data = {
        "model": "whisper-1",
        "language": language,
        "response_format": "verbose_json"
    }
    response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error in transcription: {response.status_code}, {response.text}")


def split_large_audio(audio_file_path, max_size=25 * 1024 * 1024):
    """Split audio file into chunks smaller than the specified max size."""
    audio = AudioSegment.from_file(audio_file_path, format="wav")
    chunk_duration_ms = (max_size / audio.frame_rate) / (audio.frame_width * audio.channels) * 1000
    chunks = [(audio[int(start):int(start + chunk_duration_ms)], start / 1000, (start + chunk_duration_ms) / 1000) for start in range(0, len(audio), int(chunk_duration_ms))]
    print(f"Audio file split into {len(chunks)} chunks.")
    return chunks


def split_audio_into_clips(audio_file_path, json_transcription):
    """Split audio into clips based on the start and end times from transcription."""
    print("Loading audio file...")
    audio = AudioSegment.from_file(audio_file_path, format="wav")

    print("Splitting audio into clips...")
    clips = []
    for segment in json_transcription['segments']:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        # Convert start and end times from seconds to milliseconds
        start_ms = start_time * 1000
        end_ms = end_time * 1000

        # Create a clip from the audio
        clip = audio[start_ms:end_ms]
        clips.append({
            "text": text,
            "clip": clip,
            "start": start_time,
            "end": end_time
        })
    return clips

def process_audio_chunks(audio_file_path, api_key):
    """Process a large audio file into smaller chunks and store clip metadata."""
    
    audio_chunks = split_large_audio(f"{audio_file_path}.wav")
    clip_data = []  

    for idx, chunk in tqdm(enumerate(audio_chunks), desc="whisper"):
        chunk_path = f"{audio_file_path}_chunk_{idx}.wav"
        chunk[0].export(chunk_path, format="wav")

        with open(chunk_path, "rb") as audio_chunk:
            json_transcription = transcribe_audio_with_whisper(audio_chunk, api_key)
            audio_clips = split_audio_into_clips(chunk_path, json_transcription)

            clip_data.append({
                "chunk_start": chunk[1],
                "chunk_end": chunk[2], 
                "chunk_idx": idx,
                "audio_clips": [
                    {
                        "start": clip["start"],
                        "end": clip["end"],
                    } 
                    for clip in audio_clips
                ]
            })

    metadata_file = f"{audio_file_path}_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(clip_data, f)

    print(f"Clip metadata saved to {metadata_file}")
    return metadata_file

from pydub import AudioSegment

def generate_rttm_from_file(audio_file_path, inference, treshold_closest, treshold_similarity):
    """Generate RTTM file from stored metadata."""
    
    with open(f"{audio_file_path}_metadata.json", "r") as f:
        clip_data = json.load(f)

    rttm_lines = []
    embeddings = {}
    unknow = []
    audio = AudioSegment.from_wav(f"{audio_file_path}.wav")
    idx = 0

    for entry in clip_data:
        # print(f"Processing chunk {entry['chunk_idx']}...")
        chunk_start = entry["chunk_start"]
        audio_clips = entry["audio_clips"]

        # Recreate clips from the chunk using start and end times
        formatted_clips = []
        for clip in audio_clips:
            formatted_clips.append({
                "start": clip["start"] + chunk_start,  # Global start time
                "end": clip["end"] + chunk_start,      # Global end time
                "clip": audio[(clip["start"] + chunk_start) * 1000:(clip["end"] + chunk_start) * 1000],  # Store extracted segment
                "idx": idx
            })
            idx += 1

        # Assign speakers to dynamically generated clips
        speakers, embeddings, unknow = assign_speakers_to_clips(formatted_clips, embeddings, inference, treshold_closest, treshold_similarity, unknow)
        for speaker in speakers:
            source_start = speaker["start"]
            source_end = speaker["end"]
            duration = source_end - source_start
            rttm_line = f"SPEAKER {speaker['idx']} 1 {source_start:.2f} {duration:.2f} <NA> <NA> {speaker['speaker']} {speaker['distance']}\n"
            rttm_lines.append(rttm_line)

        speakers, embeddings, unknow = assign_speakers_to_clips(unknow, embeddings, inference, treshold_closest, treshold_similarity, [])
        for speaker in speakers:
            source_start = speaker["start"]
            source_end = speaker["end"]
            duration = source_end - source_start
            rttm_line = f"SPEAKER {speaker['idx']} 1 {source_start:.2f} {duration:.2f} <NA> <NA> {speaker['speaker']} {speaker['distance']}\n"
            rttm_lines.append(rttm_line)
    for clip in unknow:
        source_start = clip["start"]
        source_end = clip["end"]
        duration = source_end - source_start
        rttm_line = f"SPEAKER {clip['idx']} 1 {source_start:.2f} {duration:.2f} <NA> {clip['speaker']} unknow {clip['distance']}\n"
        rttm_lines.append(rttm_line)   
    
    rttm_file_path = f"{audio_file_path}_{treshold_closest}_{treshold_similarity}.rttm"
    with open(rttm_file_path, "w") as rttm_file:
        rttm_file.writelines(rttm_lines)

    print(f"RTTM file saved at {rttm_file_path}")

if __name__ == "__main__":
    load_dotenv()
    audio_file_path = "../meeting/audio1725163871"
    api_key = os.getenv("OPENAI_API_KEY")
    inference = load_model(api_key)
    output_dir = f"{audio_file_path}_chunks"
    treshold_closet = 0.6
    treshold_similarity = 0.3
    
    if not os.path.exists(f'{audio_file_path}_metadata.json'):
        process_audio_chunks(audio_file_path, api_key)
    generate_rttm_from_file(audio_file_path, inference, treshold_closet, treshold_similarity)
    audio = AudioSegment.from_wav(f"{audio_file_path}.wav")
    speaker_segments = parse_rttm(f"{audio_file_path}.rttm")
    split_by_clip(speaker_segments, audio, audio_file_path)
