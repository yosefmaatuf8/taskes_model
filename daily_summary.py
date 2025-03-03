import smtplib
import json
import tiktoken
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from openai import OpenAI
from db_manager import DBManager
from globals import GLOBALS
from google_calendar_helper import GoogleCalendarHelper
from utils.utils import split_text


class DailySummary:
    def __init__(self, sender_email=GLOBALS.sender_email, sender_password=GLOBALS.sender_password,
                 smtp_server: str = "smtp.gmail.com", smtp_port: int = 587):
        load_dotenv()
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.client = OpenAI(api_key=GLOBALS.openai_api_key)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = GLOBALS.max_tokens - 500  # Buffer space
        self.from_email = self.sender_email
        self.db_manager = DBManager()

    def summarize_in_chunks(self, content: str, summary_type: str = "manager", previous_summary: str = "") -> str:
        """Summarizes large content in chunks while maintaining coherence."""
        instructions = {
            "manager": "Provide a structured summary for a Project Manager.",
            "employees": "Provide a structured summary for Employees."
        }

        prompt_template = f"""
        You are an AI meeting assistant. Generate a {summary_type} summary.

        **Instructions:** {instructions[summary_type]}

        ---
        **Previous Summary:** {previous_summary}
        **Meeting Content:** {content}
        """

        if len(content) > self.max_tokens:
            text_chunks = split_text(self.tokenizer, content, self.max_tokens - 500)
            accumulated_summary = previous_summary

            for chunk in text_chunks:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt_template + f"\n\n**Chunk:**\n{chunk}"}],
                    max_tokens=1500,
                    temperature=0.2
                )
                chunk_summary = response.choices[0].message.content.strip()
                accumulated_summary += "\n" + chunk_summary  # Append new chunk summary

            return accumulated_summary.strip()
        else:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt_template}],
                max_tokens=2000,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()

    def generate_summaries(self, structured_data, updated_topics):
        """Generates both manager and employee summaries based on meeting data."""
        meeting_content = json.dumps(structured_data, indent=2, ensure_ascii=False)
        changes_made = json.dumps(updated_topics, indent=2, ensure_ascii=False)
        combined_text = f"**Meeting Transcription:**\n{meeting_content}\n\n**Changes & Updates:**\n{changes_made}"

        manager_summary = self.summarize_in_chunks(combined_text, "manager")
        employee_summary = self.summarize_in_chunks(combined_text, "employees")

        return manager_summary, employee_summary

    def send_email(self, subject: str, body: str, recipients: list) -> None:
        """Sends summary email to recipients."""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender_email
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipients, msg.as_string())  # ✅ Passing list directly
                print(f"✅ Email sent to: {', '.join(recipients)}")

        except Exception as e:
            print(f"❌ Error sending email: {e}")

    def process_and_notify(self, structured_data, updated_topics, meeting_datetime):
        """
        Generates summaries, fetches attendees from Google Calendar, and sends summary emails.
        """
        manager_summary, employee_summary = self.generate_summaries(structured_data, updated_topics)

        # Get attendees dynamically from Google Calendar
        calendar_id = GLOBALS.calendar_id  # Update with your Google Calendar ID
        employee_emails = GoogleCalendarHelper.get_event_attendees_from_calendar(calendar_id, meeting_datetime)

        if not employee_emails:
            print("No attendees found in the calendar. Using fallback emails.")
            employee_emails = [GLOBALS.manager_email]  # Default email in case of failure

        manager_email = GLOBALS.manager_email

        # Send summaries via email
        self.send_email("Project Meeting Summary (For Manager)", manager_summary, [manager_email])
        self.send_email("Project Meeting Summary (For Employees)", employee_summary, employee_emails)


# ---- Example Usage ----
if __name__ == "__main__":
    structured_data_example = {
        "refined_transcription": [
            {"speaker": "John", "text": "We need to refactor the API.", "category": "Task"},
            {"speaker": "Alice", "text": "I completed the database migration.", "category": "Update"}
        ]
    }
    updated_topics_example = {
        "Database": [{"id": "T1", "field": "status", "new_value": "Completed"}],
        "Backend": [{"id": "T2", "field": "assigned_user", "new_value": "John"}]
    }

    meeting_datetime = "2025-02-13 10:30"  # Format: YYYY-MM-DD HH:MM

    summary_tool = DailySummary()
    summary_tool.process_and_notify(structured_data_example, updated_topics_example, meeting_datetime)
