import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os
from openai import OpenAI



class DailySummary:
    def __init__(self, sender_email: str, sender_password: str, smtp_server: str = "smtp.gmail.com",
                 smtp_port: int = 587):
        load_dotenv()
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.to_emails = []
        self.from_email = self.sender_email
        self.transcription = None

    def summarize(self, transcription: str, model: str = "gpt-3.5-turbo", max_tokens: int = 150) -> str:
        """
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",
                     "content": f"Summarize the following meeting transcription concisely:\n\n{transcription}"}
                ],
                max_tokens=max_tokens,
                temperature=0.5
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Error: Unable to generate summary."

    def send_email(self, subject: str, body: str, to_emails: list) -> None:
        """
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                for email in to_emails:
                    msg['To'] = email
                    server.sendmail(self.sender_email, email, msg.as_string())
                    print(f"Email sent to {email}")
        except Exception as e:
            print(f"Error sending email: {e}")

    def process_and_notify(self, transcription: str, to_emails: list, subject: str = "Daily Summary") -> None:
        """
        """
        summary = self.summarize(transcription)
        self.send_email(subject, summary, to_emails)


if __name__ == "__main__":
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")

    transcription = "This is a test transcription to summarize."
    to_emails = ["yosefmaatuf848@gmail.com"]

    summary_tool = DailySummary(sender_email, sender_password)
    summary_tool.process_and_notify(transcription, to_emails)
