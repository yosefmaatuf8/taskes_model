from text import extract_text_from_odt, split_text
from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Initialize tokenizer for GPT-4
tokenizer = tiktoken.encoding_for_model("gpt-4")
MAX_TOKENS = 8192  # GPT-4 limit

def parse_names(names_string, known_names):
    """Parse the names string into a dictionary of speaker labels."""
    names_dict = {}
    lines = names_string.split("\n")
    for line in lines:
        if line.strip():  # Skip empty lines
            speaker, name = line.split(":", 1)
            speaker = speaker.strip()
            name = name.strip()

            # Only add 'unknown' if the speaker is not in known_names
            if name.lower() == 'unknown' and speaker not in known_names:
                names_dict[speaker] = name
            elif name.lower() != 'unknown':
                names_dict[speaker] = name
    return names_dict

def find_in_chunk(chunks, names_string):
    messages = [
        {"role": "system", "content": "The user will have a transcript of a meeting with user separation. You are required to return who is speaker 0, etc. avraham is manager of meet, the members are: avraham, amir, shuky, benzy, yosef, yehuda, david g. and david s. The order and numbering of the speakers says nothing about their names, you have to infer names only from the transcription. If there is a speaker that you don't know, write 'unknown'."},
        {"role": "system", "content": f"These names are evaluated based on the previous parts of the conversation: {names_string}. Keep them unless you have proof to the contrary."},
        {"role": "user", "content": chunks}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=300,  # Output tasks only
            temperature=0.3
        )
        new_names = response.choices[0].message.content
        return new_names
    except Exception as e:
        return f"An error occurred during task extraction: {str(e)}"

def find_names(transcription):
    """Main function to find names from long transcriptions."""
    # Check token size of transcription
    token_count = len(tokenizer.encode(transcription))

    # Initialize the known names as an empty dictionary
    known_names = {}
    names_string = ""

    # Split transcription if needed
    if token_count > MAX_TOKENS:
        print("Transcription exceeds token limit. Splitting and summarizing...")
        chunks = split_text(transcription, MAX_TOKENS - 700)  # Leave room for prompt tokens

        # Process each chunk while keeping track of the names
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
            names_string = find_in_chunk(chunk, names_string)
            print(names_string)
            
            # Parse the string into a dictionary
            names = parse_names(names_string, known_names)

            # Update known names after each chunk
            known_names.update(names)
    else:
        names_string = find_in_chunk(transcription, names_string)
        names = parse_names(names_string, known_names)
        known_names.update(names)
    print(known_names)
    return known_names

if __name__ == "__main__":
    transcription = extract_text_from_odt("../meeting/audio1729287298.odt")
    names = find_names(transcription)

