import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os 
from dotenv import load_dotenv


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    api_key=openai_api_key 
)

def get_readable_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP errors
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        readable_text = soup.get_text(separator='\n', strip=True)
        
        return readable_text
    except requests.exceptions.RequestException as e:
        return f"Error fetching page: {e}"

def save_webpage_to_file(url, output_file="trello_api.txt"):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
        
        # with open(output_file, 'w', encoding='utf-8') as file:
        #     file.write(response.text)
        
        return response.text
     
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")


def file_to_string(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return ""


def generate_response(messages, client):
    try:
        response = client.chat.completions.create(
        model="gpt-4-32k",
        messages=messages,
        max_tokens=3000,
        n=1,
        stop=None,
        temperature=0.3
    )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"


def write_string_to_file(content, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Content successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing to {file_path}: {e}")


# url = "https://py-trello-dev.readthedocs.io/en/latest/trello.html"
# output_file = "trello_api.txt"
# systam = "youre job is to receive api wab pages and condens them, take only the function names, the partamaters and a short discription on what they do."
# messages = [{"role": "system", "content": systam}, {"role": "user", "content": get_readable_webpage(url)}]
# write_string_to_file(generate_response(messages, client), "trello_api.text")

