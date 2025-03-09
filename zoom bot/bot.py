import time
import requests
import base64


CLIENT_ID = "KZbr6bzwT7OoAFYYi3hroQ"
CLIENT_SECRET = "di16YRi3KsKSL8OpqLu4TK5s3EmMWfr7"
ACCOUNT_ID = "ULmN4UVmQ9avAxMwrGj6HA"
MEETING_ID = "your_meeting_id"  # e.g., "123456789"

def get_access_token(client_id, client_secret, account_id):
    """
    Retrieve an access token using Zoom's Server-to-Server OAuth.
    """
    auth_string = f"{client_id}:{client_secret}"
    b64_auth_string = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
    headers = {
        "Authorization": f"Basic {b64_auth_string}",
    }
    token_url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={account_id}"
    response = requests.post(token_url, headers=headers)
    if response.status_code == 200:
        token = response.json()["access_token"]
        return token
    else:
        print(f"Error getting access token: {response.status_code}, {response.text}")
        return None

def update_live_stream_settings(token, meeting_id, stream_url, stream_key, page_url):
    """
    Update the live streaming settings for a given meeting.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    url = f"https://api.zoom.us/v2/meetings/{meeting_id}/livestream"
    stream_data = {
        "stream_url": stream_url,  # Your local RTMP streaming URL
        "stream_key": stream_key,  # Your local stream key, if required
        "page_url": page_url       # Optional: a local page where your stream is displayed
    }
    response = requests.patch(url, headers=headers, json=stream_data)
    if response.status_code == 204:
        print("✅ Live streaming settings updated successfully.")
    else:
        print(f"⚠️ Failed to update streaming settings: {response.status_code} {response.text}")

def start_live_stream(token, meeting_id):
    """
    Start live streaming for the meeting.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    url = f"https://api.zoom.us/v2/meetings/{meeting_id}/livestream/status"
    data = {"action": "start"}
    response = requests.patch(url, headers=headers, json=data)
    if response.status_code == 204:
        print("✅ Live streaming started successfully.")
    else:
        print(f"⚠️ Failed to start live streaming: {response.status_code} {response.text}")

def main():
    # Define your local streaming details.
    # Ensure you have a local RTMP server running (e.g., NGINX with the RTMP module)
    stream_url = "rtmp://localhost/live"  # Local RTMP streaming endpoint
    stream_key = "local_stream_key"       # Your local stream key (if required by your server)
    page_url = "http://localhost/live"      # Optional: Local page where the stream is displayed

    # Retrieve an access token using Server-to-Server OAuth credentials
    token = get_access_token(CLIENT_ID, CLIENT_SECRET, ACCOUNT_ID)
    if token is None:
        print("Unable to get access token. Exiting.")
        return

    # Step 1: Update the live streaming settings to use your local endpoint
    update_live_stream_settings(token, MEETING_ID, stream_url, stream_key, page_url)
    
    # Optionally wait a moment to ensure settings are applied
    time.sleep(1)
    
    # Step 2: Start the live stream
    start_live_stream(token, MEETING_ID)

if __name__ == "__main__":
    main()
