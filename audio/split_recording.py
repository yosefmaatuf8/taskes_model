from pydub import AudioSegment

def parse_rttm(file) -> list:
    """Parse RTTM file into speaker segments."""
    speaker_segments = []
    with open(file, "r") as rttm:
        for line in rttm:
            parts = line.strip().split()
            if parts[0] == "SPEAKER":
                start_time, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
                end_time = start_time + duration
                speaker_segments.append([speaker, start_time, end_time])
    return speaker_segments

def split_audio(speaker_segments: list) -> None:
    # Split and save audio segments by speaker
    print("Exporting speaker-specific audio files...")
    speakers = {}
    for segment in speaker_segments:
        speaker, start_time, end_time = segment[0], segment[1], segment[2]
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
        output_file = f"{file}{safe_speaker}.mp3"
        speaker_audio.export(output_file, format="mp3", codec="mp3")
        print(f"Exported audio for {speaker} to {output_file}")

    print("Speaker segmentation completed.")

if __name__ == "__main__":
    file = "../meeting/audio1725163871"
    print("Loading audio file...")
    audio = AudioSegment.from_file(f"{file}.mp3", format="mp3")
    speaker_segments = parse_rttm(f"{file}.rttm")
    split_audio(speaker_segments)