import requests
from dotenv import load_dotenv
from trello import TrelloClient
import os

# # Load .env file
# load_dotenv()

# api_key=os.getenv('TRELLO_API_KEY'),
# api_secret=os.getenv('TRELLO_API_SECRET'),
# token=os.getenv('TRELLO_API_TOKEN')


# # 1. Get Board IDs
# boards_url = f"https://api.trello.com/1/members/me/boards?key={TRELLO_API_KEY}&token={TRELLO_API_TOKEN}"
# boards = requests.get(boards_url).json()
# print("Boards:", boards)

# # 2. Get List IDs for a Board
# board_id = "HMoEnDv2"
# lists_url = f"https://api.trello.com/1/boards/{board_id}/lists?key={api_key}&token={token}"
# lists = requests.get(lists_url).json()
# print("Lists:", lists)

# # 3. Get User IDs for a Board
# members_url = f"https://api.trello.com/1/boards/{board_id}/members?key={api_key}&token={token}"
# members = requests.get(members_url).json()
# print("Members:", members)
