import os
import time
import base64
import subprocess
import requests
from dotenv import load_dotenv

class ZoomMeetingManager:
    """Manages active Zoom meetings and starts streaming for each meeting."""
    
    ENV_DIR = "envs"
    ZOOM_API_BASE = "https://api.zoom.us/v2"

    def __init__(self):
        """Loads global API credentials and initializes tracking for active meetings."""
        load_dotenv("global.env")  # Load global API credentials
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.account_id = os.getenv("ACCOUNT_ID")
        self.active_meetings = {}

    def get_access_token(self):
        """Fetches an access token using Server-to-Server OAuth."""
        url = "https://zoom.us/oauth/token"
        params = {
            "grant_type": "account_credentials",
            "account_id": self.account_id
        }
        headers = {
            "Authorization": f"Basic {self.get_base64_credentials()}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        response = requests.post(url, data=params, headers=headers, timeout=30)

        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            print(f"⚠️ Failed to get Access Token: {response.status_code}, {response.text}")
            return None

    def get_base64_credentials(self):
        """Encodes client credentials in Base64 format for authentication."""
        credentials = f"{self.client_id}:{self.client_secret}"
        return base64.b64encode(credentials.encode()).decode()

    def get_active_meetings(self):
        """Fetches a list of currently active Zoom meetings."""
        token = self.get_access_token()
        if not token:
            return []
            pr

        url = f"{self.ZOOM_API_BASE}/users/me/meetings"
        headers = {"Authorization": f"Bearer {token}"}

        try:
            response = requests.get(url, headers=headers)
            response_data = response.json()
            if response.status_code == 200 and "meetings" in response_data:
                active_meetings = []
                for meeting in response_data["meetings"]:
                    meeting_id = meeting.get("id")
                    time.sleep(30)
                    # Query Zoom to check if this meeting is actually live
                    meeting_status_url = f"{self.ZOOM_API_BASE}/meetings/{meeting_id}"
                    status_response = requests.get(meeting_status_url, headers=headers)

                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        if status_data.get("status") == "started":
                            print(f"✅ Meeting {meeting_id} is active!")
                            active_meetings.append(meeting_id)

                return active_meetings

            return []

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network error while fetching meetings: {e}")
            return []

    def start_meeting_stream(self, meeting_id):
        """Starts the StreamRecorder for a specific meeting with its dedicated .env file."""
        env_path = os.path.join(self.ENV_DIR, f"{meeting_id}.env")
        if not os.path.exists(env_path):
            print(f"⚠️ Missing .env file for meeting {meeting_id}, skipping...")
            return None

        print(f"🚀 Starting stream for meeting {meeting_id}...")
        process = subprocess.Popen(["python", "src/stream_recorder.py"], env=dict(os.environ, DOTENV_PATH=env_path))
        return process

    def run(self):
        """Main loop that checks for active meetings and starts streaming if necessary."""
        started_meetings = set()  # 🔹 Keep track of meetings that have already started

        while True:
            active_meeting_ids = self.get_active_meetings()

            # 🔹 Start streaming only for newly detected meetings
            for meeting_id in active_meeting_ids:
                if meeting_id not in started_meetings:  # Only start once per meeting
                    print(f" Detected new active meeting {meeting_id}, starting stream...")
                    time.sleep(10)
                    self.active_meetings[meeting_id] = self.start_meeting_stream(meeting_id)
                    started_meetings.add(meeting_id)  # 🔹 Mark it as started

            # 🔹 Remove meetings that are no longer active
            for meeting_id in list(started_meetings):
                if meeting_id not in active_meeting_ids:  # If it's no longer active, remove it
                    print(f"Meeting {meeting_id} has ended, removing from tracking.")
                    started_meetings.remove(meeting_id)

            time.sleep(30)  #  Wait before checking again



if __name__ == "__main__":
    manager = ZoomMeetingManager()
    manager.run()
