import re
import json

from globals import GLOBALS
def split_text(tokenizer, text, max_tokens=GLOBALS.max_tokens):
    """Splits text into chunks ensuring it does not exceed max tokens per chunk."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        new_chunk = " ".join(current_chunk + [word])
        if len(tokenizer.encode(new_chunk)) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def extract_json_from_code_block(response_text):
    """
    Extracts JSON from a response text and attempts to fix common issues
    such as unbalanced brackets and missing closures.
    Returns a valid JSON object or None if it cannot be fixed.
    """

    # Extract JSON content from triple backticks
    pattern = r'```json(.*?)```'
    match = re.search(pattern, response_text, flags=re.DOTALL | re.IGNORECASE)
    json_str = match.group(1).strip() if match else response_text.strip()

    # Function to fix unbalanced brackets
    def fix_json_string(json_str):
        stack = []
        fixed_str = ""
        for char in json_str:
            if char in "{[":
                stack.append(char)
            elif char in "}]":
                if stack and ((char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '[')):
                    stack.pop()
                else:
                    continue  # Ignore excess closing brackets
            fixed_str += char

        while stack:
            open_bracket = stack.pop()
            fixed_str += '}' if open_bracket == '{' else ']'

        return fixed_str

    # Function to clean newlines inside JSON strings
    def clean_json_string(json_str):
        return re.sub(r"\n\s*", " ", json_str)  # Replace newlines with spaces within the JSON string

    # First attempt: clean newlines and parse as-is
    cleaned_json_str = clean_json_string(json_str)
    try:
        return json.loads(cleaned_json_str)
    except json.JSONDecodeError:
        pass

    # Second attempt: fix unbalanced brackets
    fixed_json_str = fix_json_string(cleaned_json_str)
    try:
        return json.loads(fixed_json_str)
    except json.JSONDecodeError:
        pass

    # Third attempt: append closing brackets if missing
    if not json_str.endswith('}]'):
        json_str_fixed = json_str + "}]"
        try:
            return json.loads(json_str_fixed)
        except json.JSONDecodeError:
            pass

    print("Failed to parse JSON.: ", response_text)
    return None
