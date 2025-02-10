import subprocess
import os
import wave
import time
import threading
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from tasks_manager import TasksManager
from globals import GLOBALS

class StreamRecorder:
    def __init__(self, stream_url, bucket_name, aws_access_key_id, aws_secret_access_key,
                 chunk_interval=60, silence_timeout=180, region_name='us-east-1'):
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

    def check_ffmpeg(self):
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except FileNotFoundError:
            print("Error: ffmpeg is not installed")
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
            print(f"S3 upload error: {e}")
            return None

    def get_latest_chunk_path(self):
        if not hasattr(self, 'output_dir') or not os.path.exists(self.output_dir):
            return None

        chunks = [f for f in os.listdir(self.output_dir) if f.startswith('chunk_')]
        if not chunks:
            return None

        latest_chunk = max(chunks, key=lambda x: os.path.getctime(os.path.join(self.output_dir, x)))
        return os.path.join(self.output_dir, latest_chunk)

    def cleanup_local_chunks(self):
        if hasattr(self, 'output_dir') and os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                if file.startswith('chunk_'):
                    os.remove(os.path.join(self.output_dir, file))

    def setup_new_meeting(self):
        self.meeting_start_time = datetime.now()
        self.output_dir = f"meeting_{self.meeting_start_time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.full_recording_path = os.path.join(self.output_dir, "full_meeting.wav")

    def check_audio_activity(self, audio_chunk):
        return any(abs(int.from_bytes(audio_chunk[i:i + 2], 'little', signed=True)) > self.silence_threshold
                   for i in range(0, len(audio_chunk), 2))

    def monitor_and_record(self):
        if not self.check_ffmpeg():
            return

        while True:
            if not self.check_rtmp_connection():
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

            meeting_active = False
            wav_file = None
            buffer = bytearray()
            chunk_start_time = time.time()
            last_active_audio = time.time()
            consecutive_empty_chunks = 0

            try:
                while True:
                    audio_chunk = process.stdout.read(1024 * 16)

                    if not audio_chunk:
                        consecutive_empty_chunks += 1
                        if consecutive_empty_chunks > 10:
                            print("Too many empty chunks, disconnecting...")
                            break
                        continue

                    has_activity = self.check_audio_activity(audio_chunk)

                    if has_activity and not meeting_active:
                        print("New meeting detected")
                        self.setup_new_meeting()
                        wav_file = wave.open(self.full_recording_path, 'wb')
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(16000)
                        meeting_active = True
                        buffer = bytearray()
                        chunk_start_time = time.time()

                    if has_activity:
                        last_active_audio = time.time()
                        consecutive_empty_chunks = 0

                    if meeting_active:
                        wav_file.writeframes(audio_chunk)
                        buffer.extend(audio_chunk)

                        time_since_last_activity = time.time() - last_active_audio

                        if time_since_last_activity > self.silence_timeout:
                            print("Extended silence detected, ending meeting")
                            wav_file.close()
                            s3_key = f"meetings/{self.output_dir}/full_meeting.wav"
                            s3_path = self.upload_to_s3(self.full_recording_path, s3_key)
                            if s3_path:
                                print(f"Full meeting uploaded to S3: {s3_path}")
                                tasks_manager = TasksManager()
                                tasks_manager.run(mp3_path=s3_path)

                            self.cleanup_local_chunks()
                            meeting_active = False
                            continue

                        time_since_chunk_start = time.time() - chunk_start_time
                        if time_since_chunk_start >= self.chunk_interval:
                            local_chunk_path = os.path.join(self.output_dir, f"chunk_{int(time.time())}.wav")
                            with wave.open(local_chunk_path, 'wb') as chunk_file:
                                chunk_file.setnchannels(1)
                                chunk_file.setsampwidth(2)
                                chunk_file.setframerate(16000)
                                chunk_file.writeframes(buffer)

                            buffer = bytearray()
                            chunk_start_time = time.time()

            except Exception as e:
                print(f"Recording loop error: {e}")
            finally:
                if wav_file:
                    wav_file.close()
                process.terminate()

            time.sleep(5)


if __name__ == "__main__":
    recorder = StreamRecorder(
        )

    recorder.monitor_and_record()