import re
import json

from globals import GLOBALS
def split_text(self, text, max_tokens = GLOBALS.max_tokens):
    """Split text into chunks within the token limit."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(self.tokenizer.encode(" ".join(current_chunk + [word]))) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def extract_json_from_code_block(response_text):
    """
    Searches for a JSON code block in the response text:
      ```json
      { ... }
      ```
    and returns the parsed JSON as a Python object (dict or list).

    If no JSON block is found or the JSON is invalid, tries to append "}]"
    to the end and parse again. If it still fails, returns None.
    """
    # Regex to match everything between ```json and ```
    pattern = r'```json(.*?)```'
    match = re.search(pattern, response_text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        print("No JSON code block found in the response.")
        return None

    json_str = match.group(1).strip()

    # 1) First attempt: parse as-is
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from code block: {e}")

        # 2) Second attempt: add '}]' at the end
        if not json_str.endswith('}]'):
            fix_str = json_str + "}]"
            try:
                parsed_fix = json.loads(fix_str)
                print("Successfully parsed JSON after adding '}]' at the end.")
                return parsed_fix
            except json.JSONDecodeError as e2:
                print(f"Still invalid after adding '}}]': {e2}")
                try:
                    fix_str = json_str + "}]}"
                    parsed_fix = json.loads(fix_str)
                    print("Successfully parsed JSON after adding '}]}' at the end.")
                    return parsed_fix
                except json.JSONDecodeError as e3:
                    print(f"Still invalid after adding '}}]': {e3}")

                return None
        else:
            return None
