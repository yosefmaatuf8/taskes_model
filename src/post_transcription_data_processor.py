import json
import tiktoken

from openai import OpenAI
from globals import GLOBALS
from utils.utils import extract_json_from_code_block, split_text
from db_manager.db_manager import DBManager


class TranscriptionProcessing:
    def __init__(self):
        self.client = OpenAI(api_key=GLOBALS.openai_api_key)
        self.max_tokens = GLOBALS.max_tokens - 500  # Safety margin to prevent response truncation
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
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
            prompt = f"""
            You are an AI assistant specializing in **meeting transcription analysis**.
            The following transcription is in **Hebrew**.
            Your task is to **analyze, classify, and refine** the content while ensuring the output is in **English**.
            
            **Instructions:**
            1. **Classify each sentence** into one of these categories:
               - **Task** (defines a new task).
               - **Update** (describes progress on an existing task).
               - **General** (relevant but not a task or update).
               - **Irrelevant** (not related to the discussion).

            2. **Refine and structure all classified Tasks and Updates**:
               - Identify the **speaker** and the **target audience** (if mentioned).
               - Provide a **concise English summary** of the task or update.
               - If it’s a **new task**, include its **status**.

            3. **Translate the refined transcription into English**, ensuring clarity and accuracy.
            
            ---
            **Meeting Transcription (in Hebrew, do NOT translate this section):**  
            ```  
            {chunk}  
            ```

            ---
            **Expected JSON Output:**  
            ```json
            {{
                "refined_transcription": [
                    {{
                        "speaker": "John",
                        "target": "Sarah",
                        "text": "We need to start testing the new feature",
                        "category": "Task",
                        "status": "In Progress",
                        "summary": "John informed Sarah that testing should begin for the new feature."
                    }}
                ]
            }}
            ```
            **Ensure the response is valid JSON with no additional text.**
            """

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You analyze Hebrew meeting transcriptions and return structured English output."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.2
                )

                refined_data = extract_json_from_code_block(response.choices[0].message.content)

                if refined_data:
                    refined_results.extend(refined_data.get("refined_transcription", []))
                else:
                    print("❌ Invalid response received.")

            except Exception as e:
                print(f"❌ Error processing chunk: {e}")
                continue

        print("✅ Refined Results:", refined_results)
        return {"refined_transcription": refined_results}
