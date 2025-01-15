from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken
from find_names import find_names
from text import split_text, save_to_odt, extract_text_from_odt
from diarization import diarization
from transcription import transcription_with_rttm
import json

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=openai_api_key 
)

# Initialize tokenizer for GPT-4
tokenizer = tiktoken.encoding_for_model("gpt-4")
MAX_TOKENS = 8192  # GPT-4 limit

def extract_tasks(transcription, names, list_tasks):
    """Extract tasks from the summarized transcription."""
    messages = [
        {"role": "system", "content": f"You are managing a team's task organization based on a meeting transcription and a list of existing tasks. Your roles include:\
                Creating cards for new tasks.\
                Assigning team members to tasks based on meeting context or explicit mentions.\
                Moving cards between lists to reflect task progress. There are three modes: to do, in process and done\
                Leaving comments on cards to provide updates, context, or references to the meeting discussion.\
                For each action that needs to be performed, you must create a json file that specifies the type of action and the parameters required. If no need any rule return empty string\
                names is {names}"},
        {"role": "user", "content": f"transcription: {transcription}. existing tasks: {list_tasks}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            # max_tokens=300,  # Output tasks only
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred during task extraction: {str(e)}"


def generate_tasks(transcription, list_tasks):
    """Main function to generate tasks from long transcriptions."""
    # Check token size of transcription
    names = find_names(transcription)
    transcription_tokens = len(tokenizer.encode(transcription))
    tasks_tokens = len(tokenizer.encode(json.dumps(list_tasks)))

    tasks = []

    # Split transcription if needed
    if transcription_tokens + tasks_tokens > MAX_TOKENS:
        print("Transcription exceeds token limit. Splitting and create tasks for chunks...")
        chunks = split_text(transcription, MAX_TOKENS - 1200 - tasks_tokens)  # Leave room for prompt tokens

        for i, chunk in enumerate(chunks):
            print(f"Create tasks from chunk {i + 1}/{len(chunks)}...")
            tasks_chunk = extract_tasks(chunk, names, list_tasks)
            print(tasks_chunk)
            tasks.append(tasks_chunk)

    else:
        # Extract tasks from the summarized transcription
        print("Extracting tasks from transcription...")
        tasks = extract_tasks(transcription, names, list_tasks)
    print("-" * 50)
    return tasks

if __name__ == "__main__":
    file = "../meeting/audio1729287298"
    list_task = {
        "zoom integration": {"list": "to do", "description": "", "notes": "", "members": "benzi"},
        "Asaigning names": {"list": "to do", "description": "", "notes": "", "members": "david s."},
        "Creating action items": {"list": "to do", "description": "", "notes": "", "members": "yosef"},
        "Trello api calls": {"list": "to do", "description": "", "notes": "", "members": "david g."}
    }
    if not os.path.exists(f"{file}.mp3"):
        audio = AudioSegment.from_file(f"{file}.m4a", format="m4a")
        audio.export(f"{file}.mp3", format="mp3")
    if not os.path.exists(f"{file}.rttm"):
        diarization(file)
    if not os.path.exists(f"{file}.odt"):
        result_text = transcription_with_rttm(file)
        save_to_odt(result_text, f"{file}.odt")
    else:
        result_text = extract_text_from_odt(f"{file}.odt")
    print(generate_tasks(result_text, list_task))