import datetime
from datetime import datetime, timedelta
import pytz
from google.oauth2 import service_account
from googleapiclient.discovery import build


class GoogleCalendarHelper:
    SERVICE_ACCOUNT_FILE = "/home/mefathim/PycharmProjects/taskes_model_v2/service_account.json"
    SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

    @staticmethod
    def get_event_attendees_from_calendar(calendar_id, meeting_datetime):
        """
        Retrieves attendee emails from Google Calendar for a given meeting date and time.
        """
        try:
            credentials = service_account.Credentials.from_service_account_file(
                GoogleCalendarHelper.SERVICE_ACCOUNT_FILE, scopes=GoogleCalendarHelper.SCOPES
            )
            service = build("calendar", "v3", credentials=credentials)

            # Convert meeting date string to a datetime object (assuming local timezone)
            local_tz = pytz.timezone("Asia/Jerusalem")  # Set the timezone as needed
            local_time = local_tz.localize(datetime.strptime(meeting_datetime, "%Y-%m-%d %H:%M"))

            # Allow flexibility: -2 hours before, +1 hour after
            time_min = (local_time - timedelta(hours=2)).isoformat()
            time_max = (local_time + timedelta(hours=1)).isoformat()

            events_result = service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime"
            ).execute()

            events = events_result.get("items", [])
            if not events:
                print(f"No events found at {meeting_datetime}.")
                return []

            # Extract attendees from the first matching event
            attendees = events[0].get("attendees", [])
            emails = [attendee["email"] for attendee in attendees if "email" in attendee]

            return emails

        except Exception as e:
            print(f"❌ Error fetching event attendees: {e}")
            return []
