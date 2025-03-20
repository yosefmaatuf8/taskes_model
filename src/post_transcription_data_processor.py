import json
import tiktoken

from openai import OpenAI
from globals import GLOBALS
from utils.utils import extract_json_from_code_block, split_text
from db_manager.db_manager import DBManager


class TranscriptionProcessing:
    def __init__(self):
        self.client = OpenAI(api_key=GLOBALS.openai_api_key)
        self.openai_model_name = GLOBALS.openai_model_name
        self.max_tokens_response = 16000
        self.max_tokens = GLOBALS.max_tokens - self.max_tokens_response        
        self.tokenizer = tiktoken.encoding_for_model(self.openai_model_name)
        self.db_manager = DBManager()

    def process_transcription(self, transcription_data):
        """
        Processes transcription data:
        1. Splits long transcriptions into chunks based on token limits.
        2. Classifies and refines each chunk in a single request.
        3. Ensures valid JSON output for each step.
        """

        refined_results = []
        max_token = self.max_tokens - 4750
        token_count = len(self.tokenizer.encode(transcription_data))

        # Split text into manageable chunks
        chunks = split_text(self.tokenizer, transcription_data, max_token) if token_count > max_token else [
            transcription_data]

        for chunk in chunks:
            print(chunk)
            prompt = f"""
            You are an AI assistant specializing in **meeting transcription summarization**.
            The following transcription is in **Hebrew**.
            Your task is to **summarize the key points** in **English**, merging related information into a **structured format**.

            **Instructions:**
            1. **Summarize the main discussion points** from this segment of the meeting.
            2. **Combine similar ideas into one clear statement**.
            3. **Remove irrelevant information and unrelated conversation**.
            4. **Use concise, structured English**.

            ---
            **Meeting Transcription (in Hebrew, do NOT translate this section directly, summarize instead):**  
            ```  
            {chunk}  
            ```

            ---
            **Expected JSON Output:**  
            ```json
            {{
                "summarized_transcription": [
                    {{
                        "summary": "The backend team discussed database migration issues and CI/CD optimizations. The frontend team is working on React 18 migration.",
                        "key_points": [
                            "Backend: PostgreSQL migration is ongoing, performance bottlenecks remain.",
                            "Frontend: UI framework upgrade in progress, needs design review.",
                            "DevOps: CI/CD improvements, security testing in production."
                        ]
                    }}
                ]
            }}
            ```
            **Ensure the response is valid JSON with no additional text.**
            """
             
            try:
                response = self.client.chat.completions.create(
                    model=self.openai_model_name,
                    messages=[
                        {"role": "system", "content": "You analyze Hebrew meeting transcriptions and return structured English output."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens_response,
                    temperature=0.2
                )
                refined_data = extract_json_from_code_block(response.choices[0].message.content)

                if refined_data:
                    refined_results.extend(refined_data.get("summarized_transcription", []))
                else:
                    print("❌ Invalid response received.")

            except Exception as e:
                print(f"❌ Error processing chunk: {e}")
                continue

        print("✅ Refined Results:", refined_results)
        return {"summarized_transcription": refined_results}
