import difflib
from tqdm import tqdm
import json
import os
from dotenv import load_dotenv
from globals import GLOBALS
import requests
import tiktoken

# Load environment variables from .env file
load_dotenv()
openai_api_key = GLOBALS.openai_api_key
TOKEN_LIMIT = 7200  # Max tokens per chunk (OpenAI's token limit)
OUTPUT_TOKEN_LIMIT = 3600  # Tokens allocated for the response from OpenAI

# Function to count the number of tokens in a given text
def count_tokens(text):
    """Returns the number of tokens in a given text."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

# Function to compute the total number of tokens in the prompt (system + user messages)
def compute_prompt_tokens(messages):
    """Computes the total number of tokens used in the prompt (system + user)."""
    total_tokens = 0
    for message in messages:
        total_tokens += count_tokens(message["content"])
    return total_tokens

# Function to split text into chunks based on token count
def split_into_token_chunks(text, messages, max_tokens=TOKEN_LIMIT):
    """Splits text into chunks based on token count, leaving space for response and prompt tokens."""
    # Compute tokens for the system and user message (the prompt)
    prompt_tokens = count_tokens(messages)
    
    # Calculate the remaining tokens for the actual text (response and prompt)
    remaining_tokens = max_tokens - prompt_tokens - OUTPUT_TOKEN_LIMIT

    words = text.split()
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        text_tokens = count_tokens(word)
        if current_token_count + text_tokens > remaining_tokens:
            chunks.append(" ".join(current_chunk))  # Append the current chunk if it exceeds the token limit
            current_chunk = [word]
            current_token_count = text_tokens
        else:
            current_chunk.append(word)
            current_token_count += text_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Append the last chunk

    return chunks

# Function to load text data from a JSON file
def load_text_from_json(file_path):
    """Loads transcription data from a JSON file."""
    with open(file_path, 'r', encoding="utf-8") as file:
        json_data = json.load(file)
    return [segment.get("text", "") for segment in json_data]

# Function to compare two texts and return a list of differences
def compare_texts(words1, words2):
    """Compares two lists of words and returns a list of differences using difflib."""
    return list(difflib.ndiff(words1, words2))

# Function to merge the two transcriptions based on the differences
def merge_transcript(diff):
    """Merges two transcriptions based on the differences."""
    final_text = ""
    temp_text_file1 = []  # Collect words only in the first transcription
    temp_text_file2 = []  # Collect words only in the second transcription

    for word in diff:
        if word.startswith("- "):  # Word in the first transcription but missing in the second
            temp_text_file1.append(word[2:])  

        elif word.startswith("+ "):  # Word in the second transcription but missing in the first
            temp_text_file2.append(word[2:])  

        elif word.startswith("  "):  # Word that is the same in both transcriptions
            if temp_text_file1:
                final_text += f"<<<{' '.join(temp_text_file1)}>>>"  # Merge missing words from the first transcription
                temp_text_file1 = []  
            if temp_text_file2:
                final_text += f"[[[{' '.join(temp_text_file2)}]]]"  # Merge missing words from the second transcription
                temp_text_file2 = []  
            final_text += word[2:] + " "  # Add unchanged word

    # Handle any remaining words in temp_text_file1 or temp_text_file2
    if temp_text_file1:
        final_text += f"<<<{' '.join(temp_text_file1)}>>>"
    if temp_text_file2:
        final_text += f"[[[{' '.join(temp_text_file2)}]]]"

    return final_text.strip()  # Remove trailing spaces

# Function to process and combine transcriptions
def process_and_combine(marked_text, original_segments, transcription_words):
    """Processes the diff-based chunks and sends them to GPT for merging."""
    system_message = {
        "role": "system",
        "content": "You will receive two versions of a transcription, with differences marked as follows:\n"
                    "- Unchanged text is unmarked.\n"
                    "- Text that appears only in the first transcription is wrapped in <<< >>>.\n"
                    "- Text that appears only in the second transcription is wrapped in [[[ ]]].\n\n"
                    "Your task is to merge these two transcriptions into a single, coherent version while preserving accuracy and meaning.\n"
                    "When both versions of a word or phrase differ, analyze the context to determine the most correct and natural option. If needed, combine elements from both transcriptions for the best result."
    }
    chunks = split_into_token_chunks(marked_text, str(system_message))

    combined_transcription = []

    for chunk in tqdm(chunks, desc="Combine transcriptions"):
        messages = [system_message, {"role": "user", "content": chunk}]

        # Send request to OpenAI API for merging
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={"model": "gpt-4", "messages": messages, "max_tokens": OUTPUT_TOKEN_LIMIT, "temperature": 0.7},
        )

        # Check for response errors
        if response.status_code != 200:
            print(f"Error in OpenAI response: {response.status_code}, {response.text}")
            return ""

        response_json = response.json()
        combined_transcription.append(response_json["choices"][0]["message"]["content"])

    # Join the chunks into a single string
    combined_transcription_text = " ".join(combined_transcription)
    combined_words = combined_transcription_text.split(' ')
    diff = compare_texts(combined_words, transcription_words)

    # Process the diff to align the merged transcription with the original segments
    word_index = 0
    for segment in original_segments:
        segment_text = segment['text']
        original_words_segments = segment_text.split(' ')
        original_words_index = 0

        # Temporary list to collect words for this segment
        segment_words = []

        while word_index < len(diff) and original_words_index < len(original_words_segments):
            diff_item = diff[word_index]
            diff_type = diff_item[0]  # "+" for added, "-" for removed, " " for unchanged
            word = diff_item[2:]  # The actual word (skip the diff marker)

            # Compare the word from diff with the original word
            if diff_type == " ":
                segment_words.append(word)
                original_words_index += 1
            elif diff_type == "-":
                segment_words.append(word)  # Added word from the second transcription
            elif diff_type == "+":
                original_words_index += 1  # Skip removed word
            word_index += 1

        segment['text'] = " ".join(segment_words)

    return original_segments

# Main function to process the transcriptions and save the merged result
def process_transcription(first_transcription, second_transcription, output_transcription):
    """Main function to process two transcriptions and merge them."""
    # Check if the transcription files exist
    if not os.path.exists(first_transcription):
        print(f"Error: One or both of the JSON files are missing at {first_transcription}")
        return
    
    if not os.path.exists(second_transcription):
        print(f"Error: One or both of the JSON files are missing at {second_transcription}")
        return
    
    # Load text from both transcription files
    text1 = " ".join(load_text_from_json(first_transcription))
    text2 = " ".join(load_text_from_json(second_transcription))

    # Split the texts into words
    words1 = text1.split(' ')
    words2 = text2.split(' ')

    # Compare the texts to find differences
    diff = compare_texts(words1, words2)

    # Merge the differences into a marked text
    marked_text = merge_transcript(diff)

    # Load the original segments from the second transcription
    with open(second_transcription, 'r', encoding="utf-8") as file:
        transcription_data = json.load(file)
        original_segments = transcription_data  # Assuming the JSON contains the segments as a list

    # Get the merged transcription with the original segments
    merged_transcription = process_and_combine(marked_text, original_segments, words2)

    # Save the merged transcription to a new JSON file
    with open(output_transcription, "w", encoding="utf-8") as f:
        json.dump(merged_transcription, f, ensure_ascii=False, indent=4)

    print(f"Merged transcription saved to {output_transcription}")

if __name__ == "__main__":
    # Define file paths for input and output
    file = "GMT20250203-100059_Recording"
    first_transcription = f"/home/mefathim/Documents/transcriptions/{file}/ivritai_transcription.json"
    second_transcription = f"/home/mefathim/Documents/transcriptions/{file}/whisper_transcription_{file}.json"
    output_transcription = f"/home/mefathim/Documents/transcriptions/{file}/merge.json"
    process_transcription(first_transcription, second_transcription, output_transcription)

