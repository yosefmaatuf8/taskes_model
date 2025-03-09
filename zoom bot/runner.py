import time
import requests
import subprocess
import threading
import base64
import platform
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ====== Configuration Variables ======
# Set these values in your .env file.
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
MEETING_ID = os.getenv("MEETING_ID")  # e.g., "123456789"

# RTMP and streaming settings
RTMP_ENDPOINT = os.getenv("RTMP_ENDPOINT", "test_output.mp4")
STREAM_KEY = os.getenv("STREAM_KEY", "local_stream_key")
PAGE_URL = os.getenv("PAGE_URL", "http://localhost/live")

# Create a global event for stopping threads gracefully.
stop_event = threading.Event()

# ====== Zoom API Functions ======
def get_access_token(client_id, client_secret, account_id):
    auth_string = f"{client_id}:{client_secret}"
    b64_auth_string = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
    headers = {
        "Authorization": f"Basic {b64_auth_string}",
    }
    token_url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={account_id}"
    response = requests.post(token_url, headers=headers)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print(f"Error getting access token: {response.status_code}, {response.text}")
        return None

def update_zoom_stream_settings(token, meeting_id, stream_url, stream_key, page_url):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    url = f"https://api.zoom.us/v2/meetings/{meeting_id}/livestream"
    stream_data = {
        "stream_url": stream_url,
        "stream_key": stream_key,
        "page_url": page_url
    }
    response = requests.patch(url, headers=headers, json=stream_data)
    if response.status_code == 204:
        print("✅ Zoom live streaming settings updated successfully.")
    else:
        print(f"⚠️ Failed to update streaming settings: {response.status_code} {response.text}")

def start_zoom_live_stream(token, meeting_id):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    url = f"https://api.zoom.us/v2/meetings/{meeting_id}/livestream/status"
    data = {"action": "start"}
    response = requests.patch(url, headers=headers, json=data)
    if response.status_code == 204:
        print("✅ Zoom live streaming started successfully.")
    else:
        print(f"⚠️ Failed to start live streaming: {response.status_code} {response.text}")

def get_zoom_meeting_status(token, meeting_id):
    """
    Poll the Zoom API to get the current meeting status.
    Expected statuses might be 'waiting', 'started', or 'finished' (or 'ended').
    """
    headers = {
        "Authorization": f"Bearer {token}"
    }
    url = f"https://api.zoom.us/v2/meetings/{meeting_id}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        status = response.json().get("status")
        return status
    else:
        print("Error retrieving meeting status:", response.text)
        return None

# ====== ffmpeg Streaming Functions ======
def monitor_ffmpeg_logs(process):
    while not stop_event.is_set():
        line = process.stderr.readline()
        if not line:
            break
        decoded_line = line.decode("utf-8", errors="replace")
        print(decoded_line, end="")
        if "frame=" in decoded_line or "bitrate=" in decoded_line:
            print("✅ Streaming appears to be active!")
        if "error" in decoded_line.lower():
            print("⚠️ Error detected in ffmpeg logs:", decoded_line)

def get_ffmpeg_command():
    """
    Returns an ffmpeg command list based on the operating system for capturing live audio.
    """
    system = platform.system()
    if system == "Windows":
        # Using DirectShow (update the device name as needed)
        device = "audio=CABLE Output (VB-Audio Virtual Cable)"
        command = [
            "ffmpeg",
            "-f", "dshow",
            "-i", device,
            "-c:a", "aac",
            "-b:a", "128k",
            "-f", "mp4",
            "test_output.mp4"
        ]
    elif system == "Linux":
        # Using ALSA loopback device (adjust the device string as necessary)
        device = "hw:Loopback,0,0"
        command = [
            "ffmpeg",
            "-f", "alsa",
            "-i", device,
            "-c:a", "aac",
            "-b:a", "128k",
            "-f", "flv",
            RTMP_ENDPOINT
        ]
    else:
        raise Exception(f"Unsupported OS: {system}")
    return command

def start_ffmpeg_stream():
    command = get_ffmpeg_command()
    print(f"Starting ffmpeg with command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log_thread = threading.Thread(target=monitor_ffmpeg_logs, args=(process,))
    log_thread.start()
    return process, log_thread

def monitor_zoom_meeting(token, meeting_id, ffmpeg_process):
    """
    Continuously poll the Zoom meeting status. When the meeting is no longer active,
    terminate the ffmpeg process and set the stop event.
    """
    while not stop_event.is_set():
        time.sleep(10)  # Poll every 10 seconds
        status = get_zoom_meeting_status(token, meeting_id)
        print(f"Zoom meeting status: {status}")
        if status is None:
            continue
        # Assuming that a meeting status of 'finished' or 'ended' indicates that the meeting is over.
        if status in ["finished", "ended"]:
            print("Zoom meeting has ended. Stopping ffmpeg stream.")
            ffmpeg_process.terminate()
            stop_event.set()  # Signal all threads to stop.
            break

# ====== Main Execution Flow ======
def main():
    # Step 1: Get access token from Zoom
    token = get_access_token(CLIENT_ID, CLIENT_SECRET, ACCOUNT_ID)
    if not token:
        return

    # Step 2: Update Zoom live streaming settings
    update_zoom_stream_settings(token, MEETING_ID, RTMP_ENDPOINT, STREAM_KEY, PAGE_URL)
    
    # Wait a moment to ensure settings are applied
    time.sleep(1)

    # Step 3: Start Zoom live stream
    start_zoom_live_stream(token, MEETING_ID)
    
    # Wait for Zoom to push the stream (adjust time as needed)
    time.sleep(5)
    
    # Step 4: Start ffmpeg to capture/process the audio stream from the appropriate input
    ffmpeg_process, log_thread = start_ffmpeg_stream()
    
    # Step 5: Start a separate thread to monitor the Zoom meeting status
    meeting_thread = threading.Thread(target=monitor_zoom_meeting, args=(token, MEETING_ID, ffmpeg_process))
    meeting_thread.start()
    
    try:
        # Wait for the ffmpeg process to complete.
        ffmpeg_process.wait()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Terminating streams.")
        stop_event.set()
        ffmpeg_process.terminate()
    
    # Ensure that the stop event is set so that all threads exit.
    stop_event.set()
    log_thread.join()
    meeting_thread.join()
    print("Streaming terminated.")

if __name__ == "__main__":
    main()

