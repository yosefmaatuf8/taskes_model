from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os
from pydub import AudioSegment

def diarization(file: str) -> None:
    # Load environment variables
    load_dotenv()

    # Initialize the speaker diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token=os.getenv("HUGGINGFACE_API_KEY"))

    # Apply the pipeline to the audio file
    print("Running diarization...")
    diarization = pipeline({"uri": "test_audio", "audio": f"{file}.mp3"})

    # Save the diarization output to an RTTM file
    with open(f"{file}.rttm", "w") as rttm:
        diarization.write_rttm(rttm)
    print(f"Diarization saved to {file}.rttm")

if __name__ == "__main__":
    file = "../meeting/audio1725163871"
    # audio = AudioSegment.from_file(f"{file}.m4a", format="m4a")
    # audio.export(f"{file}.mp3", format="mp3")
    diarization(file)