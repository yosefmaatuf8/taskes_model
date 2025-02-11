import subprocess
import os
import wave
import time
import threading
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from globals import GLOBALS
# Import the MeetingAnalyzer module
from transcription_handler import TranscriptionHandler



class StreamRecorder:
    def __init__(self, stream_url=GLOBALS.stream_url, bucket_name=GLOBALS.bucket_name, aws_access_key_id=GLOBALS.aws_access_key_id, aws_secret_access_key=GLOBALS.aws_secret_access_key,
                 chunk_interval=GLOBALS.chunk_interval, silence_timeout=GLOBALS.silence_timeout, region_name='us-east-1'):
        self.meeting_analyzer = None
        self.stream_url = stream_url
        self.chunk_interval = chunk_interval
        self.silence_timeout = silence_timeout
        self.silence_threshold = 100

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.bucket_name = bucket_name
        self.recording_started = False  # Flag to track if we're in an active recording

    def check_ffmpeg(self):
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except FileNotFoundError:
            print("Error: ffmpeg is not installed on the system.")
            return False

    def check_rtmp_connection(self):
        try:
            command = ['ffprobe', '-v', 'quiet', self.stream_url]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

    def upload_to_s3(self, local_file_path, s3_key):
        try:
            self.s3_client.upload_file(local_file_path, self.bucket_name, s3_key)
            return f"s3://{self.bucket_name}/{s3_key}"
        except ClientError as e:
            print(f"Error uploading to S3: {e}")
            return None

    def cleanup_local_files(self):
        if hasattr(self, 'output_dir') and os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                if file.startswith('chunk_') or file == "full_meeting.wav":
                    os.remove(os.path.join(self.output_dir, file))
            os.rmdir(self.output_dir)

    def setup_new_meeting(self):
        self.meeting_start_time = datetime.now()
        self.output_dir = f"meeting_{self.meeting_start_time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.full_recording_path = os.path.join(self.output_dir, "full_meeting.wav")
        self.meeting_analyzer = TranscriptionHandler(self.output_dir)

    def check_audio_activity(self, audio_chunk):
        if not audio_chunk:
            return False
        return any(abs(int.from_bytes(audio_chunk[i:i + 2], 'little', signed=True)) > self.silence_threshold
                   for i in range(0, len(audio_chunk), 2))

    def monitor_and_record(self):
        if not self.check_ffmpeg():
            return

        while True:
            if not self.check_rtmp_connection():
                print("Waiting for stream connection...")
                time.sleep(5)
                continue

            command = [
                "ffmpeg",
                "-i", self.stream_url,
                "-ac", "1",
                "-ar", "16000",
                "-f", "wav",
                "-"
            ]

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            stderr_thread = threading.Thread(target=lambda: process.stderr.read())
            stderr_thread.daemon = True
            stderr_thread.start()

            wav_file = None
            last_active_audio = None
            consecutive_silence_chunks = 0
            audio_activity_detected = False
            buffer = bytearray()
            chunk_start_time = time.time()

            try:
                while True:
                    audio_chunk = process.stdout.read(1024 * 16)

                    if not audio_chunk:
                        time.sleep(0.1)
                        continue

                    has_activity = self.check_audio_activity(audio_chunk)

                    if has_activity:
                        consecutive_silence_chunks = 0
                        if not self.recording_started and not audio_activity_detected:
                            audio_activity_detected = True
                            continue

                        if audio_activity_detected and not self.recording_started:
                            print("Meeting detected, starting recording...")
                            self.setup_new_meeting()
                            wav_file = wave.open(self.full_recording_path, 'wb')
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(16000)
                            self.recording_started = True
                            last_active_audio = time.time()
                    else:
                        consecutive_silence_chunks += 1

                    if self.recording_started:
                        wav_file.writeframes(audio_chunk)
                        buffer.extend(audio_chunk)

                        if has_activity:
                            last_active_audio = time.time()
                        elif time.time() - last_active_audio > self.silence_timeout:
                            print("Meeting ended, saving recording...")
                            wav_file.close()
                            file_size = os.path.getsize(self.full_recording_path)
                            if file_size > 1024 * 100:
                                s3_key = f"meetings/{self.output_dir}/full_meeting.wav"
                                s3_path = self.upload_to_s3(self.full_recording_path, s3_key)
                                if s3_path:
                                    print(f"Meeting uploaded to S3: {s3_path}")
                            self.cleanup_local_files()
                            self.recording_started = False
                            audio_activity_detected = False
                            break

                        if time.time() - chunk_start_time >= self.chunk_interval:
                            chunk_path = os.path.join(self.output_dir, f"chunk_{int(time.time())}.wav")
                            with wave.open(chunk_path, 'wb') as chunk_file:
                                chunk_file.setnchannels(1)
                                chunk_file.setsampwidth(2)
                                chunk_file.setframerate(16000)
                                chunk_file.writeframes(buffer)
                            buffer = bytearray()
                            chunk_start_time = time.time()
                            self.meeting_analyzer.run(chunk_path, chunk_start_time,
                                                                time.time())

                    if consecutive_silence_chunks > 50 and not self.recording_started:
                        audio_activity_detected = False

            except Exception as e:
                print(f"Error during recording: {e}")
            finally:
                if wav_file:
                    wav_file.close()
                process.terminate()

            time.sleep(5)


if __name__ == "__main__":
    recorder = StreamRecorder(
        )

    recorder.monitor_and_record()