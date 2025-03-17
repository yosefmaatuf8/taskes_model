import requests
import json

def send_to_api(api_url, text, max_length=100):
    """Sends a POST request to the API and returns the generated text."""
    try:
        data = {"text": text, "max_length": max_length}
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, headers=headers, data=json.dumps(data))

        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        result = response.json()
        return result["generated_text"]

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return None
    except KeyError as e:
        print(f"Error: missing key in response: {e}")
        return None

# Example usage:
aws_api = "" # insert your public api instance
api_url = f"http://{aws_api}/generate/"
input_text = "how i make pasta?"
generated_text = send_to_api(api_url, input_text)

if generated_text:
    print("Generated Text:", generated_text)
else:
    print("Failed to get generated text.")