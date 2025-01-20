from pydub import AudioSegment
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv

class AudioHandler:
    def __init__(self, huggingface_api_key=None):
        load_dotenv()
        self.api_key = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",
            use_auth_token=self.api_key
        )

    def convert_to_mp3(self, file_path):
        audio = AudioSegment.from_file(file_path, format="m4a")
        mp3_path = file_path.replace(".m4a", ".mp3")
        audio.export(mp3_path, format="mp3")
        return mp3_path

    def run_diarization(self, file_path):
        digitization = self.pipeline({"uri": "audio", "audio": file_path})
        rttm_path = file_path.replace(".mp3", ".rttm")
        with open(rttm_path, "w") as rttm_file:
            digitization.write_rttm(rttm_file)
        return rttm_path

    def split_audio(self, audio, max_size_mb):
        file_size_mb = len(audio) / (1024 * 1024)
        if file_size_mb <= max_size_mb:
            return [audio]

        parts = []
        part_duration = len(audio) // (file_size_mb // max_size_mb + 1)
        for start in range(0, len(audio), part_duration):
            parts.append(audio[start:start + part_duration])
        return parts


    # Other existing methods remain as before...
    # def transcribe_audio(self, audio_segment):
    #     buffer = BytesIO()
    #     audio_segment.export(buffer, format="mp3")
    #     buffer.seek(0)
    #
    #     url = "https://api.openai.com/v1/audio/transcriptions"
    #     headers = {"Authorization": f"Bearer {self.api_key}"}
    #     files = {"file": ("audio.mp3", buffer, "audio/mp3")}
    #     data = {"model": "whisper-1", "language": self.language}
    #
    #     response = requests.post(url, headers=headers, files=files, data=data)
    #     if response.status_code == 200:
    #         return response.json()["text"]
    #     else:
    #         raise Exception(f"Error in transcription: {response.status_code}, {response.text}")
    #
    # def transcription_with_rttm(self, file_path, max_size_mb=24.5):
    #     # Convert audio if necessary
    #     mp3_path = file_path if file_path.endswith(".mp3") else self.audio_handler.convert_to_mp3(file_path)
    #     rttm_path = mp3_path.replace(".mp3", ".rttm")
    #     if not os.path.exists(rttm_path):
    #         self.audio_handler.run_diarization(mp3_path)
    #
    #     # Parse diarization segments
    #     with open(rttm_path, "r") as rttm_file:
    #         speaker_segments = [
    #             line.strip().split() for line in rttm_file.readlines() if not line.startswith("#")
    #         ]
    #
    #     audio = AudioSegment.from_file(mp3_path, format="mp3")
    #     conversation = ""
    #     prev_speaker = None
    #     combined_segment = AudioSegment.empty()
    #
    #     for segment in speaker_segments:
    #         speaker, start_time, end_time = segment[0], float(segment[3]), float(segment[4])
    #         if speaker == prev_speaker:
    #             combined_segment += audio[start_time * 1000:end_time * 1000]
    #         else:
    #             if combined_segment.duration_seconds > 0:
    #                 transcription_result = self.transcribe_audio(combined_segment)
    #                 conversation += f"{prev_speaker}: {transcription_result.strip()}\n\n"
    #
    #             prev_speaker = speaker
    #             combined_segment = audio[start_time * 1000:end_time * 1000]
    #
    #     # Process the last segment
    #     if combined_segment.duration_seconds > 0:
    #         transcription_result = self.transcribe_audio(combined_segment)
    #         conversation += f"{prev_speaker}: {transcription_result.strip()}\n"
    #
    #     return conversation

    # def process_transcription(self, file_path, max_size_mb=24.5):
    #     audio = AudioSegment.from_file(file_path, format="mp3")
    #     audio_parts = self.audio_handler.split_audio(audio, max_size_mb)
    #     return " ".join(self.transcribe_audio(part) for part in audio_parts)

