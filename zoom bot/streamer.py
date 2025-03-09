import time
import requests
import subprocess
import threading
import base64

# Replace these with your actual credentials and meeting details for the Server-to-Server OAuth app
CLIENT_ID = "KZbr6bzwT7OoAFYYi3hroQ"
CLIENT_SECRET = "di16YRi3KsKSL8OpqLu4TK5s3EmMWfr7"
ACCOUNT_ID = "ULmN4UVmQ9avAxMwrGj6HA"
MEETING_ID = "your_meeting_id"  # e.g., "123456789"

# RTMP and streaming settings
RTMP_ENDPOINT = "rtmp://localhost/live/stream"
STREAM_KEY = "local_stream_key"      # If needed
PAGE_URL = "http://localhost/live"     # Optional

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

def monitor_ffmpeg_logs(process):
    while True:
        line = process.stderr.readline()
        if not line:
            break
        decoded_line = line.decode("utf-8", errors="replace")
        print(decoded_line, end="")
        if "frame=" in decoded_line or "bitrate=" in decoded_line:
            print("✅ Streaming appears to be active!")
        if "error" in decoded_line.lower():
            print("⚠️ Error detected in ffmpeg logs:", decoded_line)

def start_ffmpeg_stream():
    command = [
        "ffmpeg",
        "-re",
        "-i", "sample.wav",  # Replace with your actual audio input or device if capturing live audio
        "-c:a", "aac",
        "-b:a", "128k",
        "-f", "flv",
        RTMP_ENDPOINT
    ]
    print(f"Starting ffmpeg with command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log_thread = threading.Thread(target=monitor_ffmpeg_logs, args=(process,))
    log_thread.start()
    process.wait()
    log_thread.join()

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
    
    # Step 4: Start ffmpeg to capture/process the stream from the RTMP endpoint
    start_ffmpeg_stream()

if __name__ == "__main__":
    main()
