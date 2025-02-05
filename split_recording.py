from pydub import AudioSegment
import os

def parse_rttm(file) -> list:
    """Parse RTTM file into speaker segments."""
    speaker_segments = []
    with open(file, "r") as rttm:
        for line in rttm:
            parts = line.strip().split()
            if parts[0] == "SPEAKER":
                idx, start_time, duration, speaker, distance = parts[1], float(parts[3]), float(parts[4]), parts[7], parts[8]
                end_time = start_time + duration
                speaker_segments.append([idx, speaker, start_time, end_time, distance])
    return speaker_segments

def split_audio(speaker_segments: list, audio, file) -> None:
    # Split and save audio segments by speaker
    print("Exporting speaker-specific audio files...")
    speakers = {}
    for segment in speaker_segments:
        idx, speaker, start_time, end_time = segment[0], segment[1], segment[2], segment[3]
        speaker_audio = AudioSegment.empty()
        if speaker in speakers:
            speakers[speaker] += [[start_time, end_time]]
        else:
            speakers[speaker] = [[start_time, end_time]]
       
    for speaker, segments in speakers.items():
        speaker_audio = AudioSegment.empty()
        for start_time, end_time in segments:
            segment = audio[start_time * 1000:end_time * 1000]  # Convert seconds to milliseconds
            speaker_audio += segment

        
        # Save the speaker-specific audio in M4A format
        safe_speaker = speaker.replace(" ", "_").replace("/", "-")  # Sanitize speaker name for filenames
        output_file = f"{file}{safe_speaker}.wav"
        speaker_audio.export(output_file, format="wav")
        print(f"Exported audio for {speaker} to {output_file}")

    print("Speaker segmentation completed.")

def split_by_clip(speaker_segments: list, audio, file):
    for segment in speaker_segments:
        idx, speaker, start_time, end_time, distance = segment[0], segment[1], segment[2], segment[3], segment[4]
        speaker_audio = audio[start_time * 1000:end_time * 1000]
        output_dir = f"{file}/{speaker}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = f"{output_dir}/{idx}"
        speaker_audio.export(output_file, format="wav")

if __name__ == "__main__":
    file = "../meeting/audio1725163871"
    print("Loading audio file...")
    audio = AudioSegment.from_wav(f"{file}.wav")
    speaker_segments = parse_rttm(f"{file}_truth.rttm")
    split_by_clip(speaker_segments, audio, file)