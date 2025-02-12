from trello import TrelloClient
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Initialize Trello Client
client = TrelloClient(
    api_key=os.getenv('TRELLO_API_KEY'),
    api_secret=os.getenv('TRELLO_API_SECRET'),
    token=os.getenv('TRELLO_API_TOKEN')
)

# Replace these with actual IDs from your environment
BOARD_ID = os.getenv('TRELLO_BOARD_ID')

# Function to get user ID by username
def get_user_id_by_username(board_id, username):
    """Fetches the user ID from a username."""
    board = client.get_board(board_id)
    members = board.get_members()
    for member in members:
        if member.username == username:
            return member.id
    raise ValueError(f"User '{username}' not found on board with ID '{board_id}'.")

# Function to get list ID by list name
def get_list_id_by_name(board_id, list_name):
    """Fetches the list ID from a list name."""
    board = client.get_board(board_id)
    lists = board.all_lists()
    for trello_list in lists:
        if trello_list.name == list_name:
            return trello_list.id
    raise ValueError(f"List '{list_name}' not found on board with ID '{board_id}'.")

# Function to get a card by its name
def get_card_by_name(board_id, card_name):
    """Fetches a card object by its name."""
    board = client.get_board(board_id)
    for trello_list in board.all_lists():
        for card in trello_list.list_cards():
            if card.name == card_name:
                return card
    raise ValueError(f"Card '{card_name}' not found on board with ID '{board_id}'.")

# Function to create a new card in a list
def create_card(list_name, card_name, card_description):
    """Creates a new card in the specified list by name."""
    list_id = get_list_id_by_name(BOARD_ID, list_name)
    board = client.get_board(BOARD_ID)
    trello_list = board.get_list(list_id)
    card = trello_list.add_card(card_name, card_description)
    print(f"Card '{card_name}' created in list '{list_name}'.")
    return card

# Function to assign a member to a card
def assign_member_to_card(card_name, username):
    """Assigns a user to the specified card using their username."""
    card = get_card_by_name(BOARD_ID, card_name)
    user_id = get_user_id_by_username(BOARD_ID, username)
    card.assign(user_id)
    print(f"Assigned user '{username}' to card '{card_name}'.")

# Function to move a card between lists
def move_card_to_list(card_name, target_list_name):
    """Moves a card to the specified list by name."""
    card = get_card_by_name(BOARD_ID, card_name)
    target_list_id = get_list_id_by_name(BOARD_ID, target_list_name)
    card.change_list(target_list_id)
    print(f"Moved card '{card_name}' to list '{target_list_name}'.")

# Function to comment on a card
def comment_on_card(card_name, comment):
    """Adds a comment to the specified card by name."""
    card = get_card_by_name(BOARD_ID, card_name)
    card.comment(comment)
    print(f"Added comment on card '{card_name}': {comment}")

# Function to get all card details
def get_card_details():
    """Fetches all card details including users, description, and list information."""
    board = client.get_board(BOARD_ID)
    cards_details = {}

    for trello_list in board.all_lists():
        for card in trello_list.list_cards():
            # Fetch member usernames using 'idMembers'
            card_members = [client.get_member(member_id).username for member_id in card.member_ids]

            cards_details[card.name] = {
                "users": card_members,
                "description": card.desc,
                "list": trello_list.name
            }

    return cards_details


def get_all_card_details():
    """Fetches all available card details including users, description, list info, due dates, labels, badges, checklists, attachments, etc."""
    board = client.get_board(BOARD_ID)
    cards_details = {}

    for trello_list in board.all_lists():
        for card in trello_list.list_cards():
            # Get full member info (you can adjust which member details you need)
            card_members = []
            for member_id in card.member_ids:
                member = client.get_member(member_id)
                card_members.append({
                    "id": member.id,
                    "username": member.username,
                    "fullName": getattr(member, 'fullName', None)  # if available
                })

            # Get label information (id, name, color)
            labels = []
            # Assuming card.labels is a list of label objects
            for label in card.labels:
                labels.append({
                    "id": label.id,
                    "name": label.name,
                    "color": label.color
                })

            # Get checklists data (if available)
            checklists = []
            try:
                # Many libraries provide a fetch_checklists() method
                for checklist in card.fetch_checklists():
                    checklist_items = []
                    for item in checklist.items:
                        checklist_items.append({
                            "id": item.get("id"),
                            "name": item.get("name"),
                            "state": item.get("state")
                        })
                    checklists.append({
                        "id": checklist.id,
                        "name": checklist.name,
                        "items": checklist_items
                    })
            except Exception:
                # If checklists are not supported or an error occurs, skip
                pass

            # Get attachments data (if available)
            attachments = []
            if hasattr(card, 'attachments'):
                for attachment in card.attachments:
                    attachments.append({
                        "id": attachment.get("id"),
                        "name": attachment.get("name"),
                        "url": attachment.get("url")
                    })

            # Build the complete details dictionary for the card.
            # You can add more fields here if needed.
            cards_details[card.id] = {
                "name": card.name,
                "description": card.desc,
                "list": trello_list.name,
                "members": card_members,
                "labels": labels,
                "due": getattr(card, 'due', None),
                "dueComplete": getattr(card, 'due_complete', None),
                "dateLastActivity": getattr(card, 'dateLastActivity', None),
                "closed": getattr(card, 'closed', None),
                "url": getattr(card, 'shortUrl', None),
                "badges": getattr(card, 'badges', None),
                "idBoard": getattr(card, 'idBoard', None),
                "idList": getattr(card, 'idList', None),
                "checklists": checklists,
                "attachments": attachments,
                # If there are other attributes on the card that you need,
                # you can add them here.
            }

    return cards_details

# Run the example flow
if __name__ == "__main__":
    # comment_on_card('Creating action items', 'i got to much comments')
    print(get_card_details())


