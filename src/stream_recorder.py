import subprocess
import os
import wave
import time
import select
import threading
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from globals import GLOBALS
from src.main_manager import Manager  # Real-time transcription


class StreamRecorder:
    def __init__(self, stream_url=GLOBALS.stream_url, bucket_name=GLOBALS.bucket_name,
                 aws_access_key_id=GLOBALS.aws_access_key_id, aws_secret_access_key=GLOBALS.aws_secret_access_key,
                 chunk_interval=GLOBALS.chunk_interval, silence_timeout=GLOBALS.silence_timeout, region_name='us-east-1'):
        self.meeting_start_time = None
        self.meeting_start_time_peth = None
        self.last_chunk_time = None  # Not used for chunk timing anymore
        self.last_recorded_duration = 0  # Tracks the actual recorded audio duration (in seconds)
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
                if file.endswith('.wav'):
                    os.remove(os.path.join(self.output_dir, file))
            if not os.listdir(self.output_dir):
                os.rmdir(self.output_dir)

    def setup_new_meeting(self):
        self.meeting_start_time = time.monotonic()
        self.last_chunk_time = time.monotonic()  # Not used for chunk timing anymore
        self.last_recorded_duration = 0  # Reset recorded duration
        self.meeting_start_time_peth = datetime.now()
        self.output_dir = GLOBALS.output_path + f"/meeting_{self.meeting_start_time_peth.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.full_recording_path = os.path.join(self.output_dir, "full_meeting.wav")
        self.meeting_analyzer = Manager(self.full_recording_path, self.output_dir)

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
            last_active_audio = time.time()
            consecutive_silence_chunks = 0
            audio_activity_detected = False
            buffer = bytearray()
            silence_counter = 0
            stream_lost_start_time = None

            try:
                while True:
                    # בדיקה אם `ffmpeg` נסגר
                    if process.poll() is not None:
                        if stream_lost_start_time is None:
                            stream_lost_start_time = time.time()  # מתחילים למדוד כמה זמן הזרם כבוי
                        elif time.time() - stream_lost_start_time >= self.silence_timeout:
                            print("FFmpeg process closed for too long. Stopping recording...")
                            break  # זרם הפסיק לזמן ממושך - מפסיקים את ההקלטה
                        time.sleep(1)
                        continue  # ממתינים לזרם לחזור

                    audio_chunk = process.stdout.read(1024 * 16)
                    if not audio_chunk:
                        silence_counter += 1
                        if stream_lost_start_time is None:
                            stream_lost_start_time = time.time()
                        time.sleep(1)

                        if time.time() - stream_lost_start_time >= self.silence_timeout:
                            print("No audio detected for too long. Stopping recording...")
                            break

                        continue

                        # אם חזר אודיו - מאפסים את זמן איבוד הזרם
                    stream_lost_start_time = None
                    silence_counter = 0

                    has_activity = self.check_audio_activity(audio_chunk)

                    if has_activity:
                        consecutive_silence_chunks = 0
                        last_active_audio = time.time()
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

                    else:
                        consecutive_silence_chunks += 1

                    if self.recording_started:
                        wav_file.writeframes(audio_chunk)
                        buffer.extend(audio_chunk)

                        if has_activity:
                            last_active_audio = time.time()
                        elif time.time() - last_active_audio > self.silence_timeout:
                            print("Meeting ended, saving recording...")
                            break

                        frames_written = wav_file.tell()
                        recorded_duration = frames_written / 16000.0

                        if recorded_duration - self.last_recorded_duration >= self.chunk_interval:
                            chunk_start_time = self.last_recorded_duration
                            chunk_end_time = recorded_duration
                            self.meeting_analyzer.ran_for_chunks(chunk_start_time, chunk_end_time)
                            self.last_recorded_duration = recorded_duration

                    if consecutive_silence_chunks > 50 and not self.recording_started:
                        audio_activity_detected = False

            except Exception as e:
                print(f"Error during recording: {e}")

            finally:
                if self.recording_started:
                    print("Meeting ended, saving recording...")
                    if wav_file:
                        wav_file.close()
                    file_size = os.path.getsize(self.full_recording_path) if os.path.exists(
                        self.full_recording_path) else 0
                    if file_size > 1024 * 100:
                        self.meeting_analyzer.stop_transcription(self.meeting_start_time_peth)
                        s3_key = f"meetings/{self.output_dir}/full_meeting.wav"
                        s3_path = self.upload_to_s3(self.full_recording_path, s3_key)
                        if s3_path:
                            print(f"Meeting uploaded to S3: {s3_path}")
                    self.cleanup_local_files()
                    self.recording_started = False
                    audio_activity_detected = False

                # ודא שהתהליך נסגר
                process.terminate()
                process.wait()

            time.sleep(5)


if __name__ == "__main__":
    recorder = StreamRecorder()
    recorder.monitor_and_record()
