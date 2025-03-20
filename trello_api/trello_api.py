import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import requests
import json
from dotenv import load_dotenv
from trello import TrelloClient
from globals import GLOBALS

class TrelloAPI:
    def __init__(self, trello_board_id=None):
        load_dotenv()
        self.client = TrelloClient(
            api_key=os.getenv('TRELLO_API_KEY'),
            api_secret=os.getenv('TRELLO_API_SECRET'),
            token=os.getenv('TRELLO_API_TOKEN')
        )
        self.board_id = trello_board_id or GLOBALS.bord_id_manager
        self.base_url = "https://api.trello.com/1"
        self.api_key = os.getenv('TRELLO_API_KEY')
        self.token = os.getenv('TRELLO_API_TOKEN')
        self.custom_fields = self.get_all_custom_fields()


    def get_list_id_by_name(self, list_name):
            """Fetch list ID by name."""
            board = self.client.get_board(self.board_id)
            for trello_list in board.all_lists():
                if  trello_list.name and  str(trello_list.name).lower() == str(list_name).lower():
                    return trello_list.id
            return None

    def get_card_by_name(self, card_name):
        """Fetch a card by its name."""
        board = self.client.get_board(self.board_id)
        for trello_list in board.all_lists():
            for card in trello_list.list_cards():
                if card.name and  str(card.name).lower() == str(card_name).lower():
                    return card
        return None


    def get_all_custom_fields(self):
        """Fetch all custom fields in the board and store them in a dictionary."""
        url = f"{self.base_url}/boards/{self.board_id}/customFields?key={self.api_key}&token={self.token}"
        response = requests.get(url)

        if response.status_code == 200:
            custom_fields = response.json()
            return {field["name"]: field["id"] for field in custom_fields}
        else:
            print(f"❌ Failed to retrieve custom fields: {response.text}")
            return {}

    def create_card(self, list_name, card_name, card_description, task_id, status, assigned_user):
        """Create a new card in the specified list with relevant details inside the description."""
        list_id = self.get_list_id_by_name(list_name)
        if not list_id:
            list_id = self.create_list(list_name).id

        trello_list = self.client.get_list(list_id)
        existing_card = self.get_card_by_name(card_name)
        if existing_card:
            print(f"Card '{card_name}' already exists.")
            return existing_card.id

        full_description = f"**Task ID:** {task_id}\n**Status:** {status}\n**Assigned User:** {assigned_user}\n\n{card_description}"

        card = trello_list.add_card(card_name, full_description)
        print(f"Card '{card_name}' created in list '{list_name}'.")
        return card.id

    def parse_existing_description(self, description):
        """
        Extracts the existing values from the description.
        Returns a dictionary of existing values.
        """
        existing_data = {
            "task_id": None,
            "status": None,
            "assigned_user": None,
            "additional_info": ""
        }

        pattern = re.compile(r"\*\*Task ID:\*\* (.*?)\n\*\*Status:\*\* (.*?)\n\*\*Assigned User:\*\* (.*?)\n\n(.*)",
                             re.DOTALL)
        match = pattern.match(description.strip())

        if match:
            existing_data["task_id"] = match.group(1).strip()
            existing_data["status"] = match.group(2).strip()
            existing_data["assigned_user"] = match.group(3).strip()
            existing_data["additional_info"] = match.group(4).strip()
        else:
            existing_data["additional_info"] = description.strip()  # אם אין פורמט ידוע, נשמור את כל התיאור

        return existing_data

    def update_card(self, card_name, task_id=None, new_status=None, new_assigned_user=None, new_description=None):
        """Update card details including status, assigned user, and description while keeping existing values."""
        card = self.get_card_by_name(card_name)
        if not card:
            print(f"❌ Card '{card_name}' not found.")
            return

        existing_data = self.parse_existing_description(card.desc or "")

        updated_description = f"""
           **Task ID:** {task_id or existing_data["task_id"]}
           **Status:** {new_status or existing_data["status"]}
           **Assigned User:** {new_assigned_user or existing_data["assigned_user"]}

           {new_description or existing_data["additional_info"]}
           """.strip()

        card.set_description(updated_description)
        print(f"✅ Updated card '{card_name}'.")

    def create_custom_field(self, field_name, field_type="text"):
        """Create a new custom field on the board."""
        url = f"{self.base_url}/customFields"
        payload = {
            "idModel": self.board_id,
            "modelType": "board",
            "name": field_name,
            "type": field_type,
            "key": self.api_key,
            "token": self.token
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            custom_field_id = response.json().get("id")
            print(f"✅ Custom field '{field_name}' created with ID {custom_field_id}")
            self.custom_fields[field_name] = custom_field_id  # Store it locally
            return custom_field_id
        else:
            print(f"❌ Failed to create custom field '{field_name}': {response.text}")
            return None

    def update_custom_field(self, card_id, field_name, field_value):
        """Update or create a custom field on a specific card."""
        field_id = self.custom_fields.get(field_name)

        if not field_id:
            print(f"⚠️ Custom field '{field_name}' not found. Creating it...")
            field_id = self.create_custom_field(field_name)

            if not field_id:
                print(f"❌ Failed to create custom field '{field_name}'. Skipping update.")
                return False

        url = f"{self.base_url}/cards/{card_id}/customField/{field_id}/item"
        payload = {
            "value": {"text": field_value},
            "key": self.api_key,
            "token": self.token
        }
        response = requests.put(url, json=payload)

        if response.status_code == 200:
            print(f"✅ Updated custom field '{field_name}' on card '{card_id}' with value: {field_value}")
            return True
        else:
            print(f"❌ Failed to update custom field '{field_name}' on card '{card_id}': {response.text}")
            return False
    def get_user_id_by_username(self, username):
        """Fetch user ID by username."""
        board = self.client.get_board(self.board_id)
        members = board.get_members()
        for member in members:
            if member.username == username:
                return member.id
        return None


    def create_list(self, list_name):
        """Create a new list for a topic if it doesn't exist."""
        board = self.client.get_board(self.board_id)
        if not self.get_list_id_by_name(list_name):
            trello_list = board.add_list(list_name)
            print(f"List '{list_name}' created.")
            return trello_list
        print(f"List '{list_name}' already exists.")
        return None


    def delete_list(self, list_name):
        """Delete a list by name if it exists."""
        list_id = self.get_list_id_by_name(list_name)
        if list_id:
            trello_list = self.client.get_list(list_id)
            trello_list.close()
            print(f"List '{list_name}' deleted.")
        else:
            print(f"List '{list_name}' not found.")

    def delete_card(self, card_name):
        """Delete a card by name if it exists."""
        card = self.get_card_by_name(card_name)
        if card:
            card.delete()
            print(f"Card '{card_name}' deleted.")
        else:
            print(f"Card '{card_name}' not found.")

    def move_card_to_list(self, card_name, target_list_name):
        """Move a card to the specified list by name."""
        card = self.get_card_by_name(card_name)
        target_list_id = self.get_list_id_by_name(target_list_name)
        if card and target_list_id:
            card.change_list(target_list_id)
            print(f"Moved card '{card_name}' to list '{target_list_name}'.")
        else:
            print(f"Card or target list not found.")

    def assign_member_to_card(self, card_name, username):
        """Assign a user to a card."""
        card = self.get_card_by_name(card_name)
        user_id = self.get_user_id_by_username(username)
        if card and user_id:
            card.assign(user_id)
            print(f"Assigned user '{username}' to card '{card_name}'.")
        else:
            print(f"Card or user not found.")

    def remove_member_from_card(self, card_name, username):
        """Remove a user from a card."""
        card = self.get_card_by_name(card_name)
        user_id = self.get_user_id_by_username(username)
        if not card:
            print(f"Card '{card_name}' not found.")
            return
        if not user_id:
            print(f"User '{username}' not found.")
            return
        if user_id not in card.member_ids:
            print(f"User '{username}' is not assigned to card '{card_name}'.")
            return
        card.remove_member(user_id)
        print(f"Removed user '{username}' from card '{card_name}'.")

    def comment_on_card(self, card_name, comment):
        """Add a comment to a card."""
        card = self.get_card_by_name(card_name)
        if card:
            card.comment(comment)
            print(f"Added comment on card '{card_name}': {comment}")
        else:
            print(f"Card '{card_name}' not found.")

    def get_all_card_details(self):
        """
        Fetch all card details from OPEN lists only, including users, description, and list information.
        """
        board = self.client.get_board(self.board_id)
        cards_details = {}

        for trello_list in board.list_lists("open"):  # Fetch only active (not archived) lists
            for card in trello_list.list_cards():
                card_members = [self.client.get_member(member_id).username for member_id in card.member_ids]
                cards_details[card.name] = {
                    "id": card.id,
                    "users": card_members,
                    "description": card.desc,
                    "list": trello_list.name
                }
        return cards_details


    def get_usernames(self):
        """Fetch all usernames of members in the board."""
        board = self.client.get_board(self.board_id)
        members = board.get_members()
        return [member.username for member in members]

    def get_id_and_usernames(self):
        """Fetch all user IDs and usernames from the board."""
        board = self.client.get_board(self.board_id)
        members = board.get_members()
        return [(member.id, member.username, member.full_name) for member in members]

    def get_all_boards(self):
        """Retrieve all boards associated with the authenticated Trello user."""
        boards = self.client.list_boards()
        return [{"id": board.id, "name": board.name, "url": board.url} for board in boards]

    def clear_board_and_lists(self):
        """
        Deletes all cards and lists from the Trello board.
        """
        board = self.client.get_board(self.board_id)
        # Now, delete all lists (Trello does not allow direct deletion, so we close them)
        for trello_list in board.all_lists():
            trello_list.close()  # Close list (equivalent to deleting in Trello)
            print(f"List '{trello_list.name}' closed.")

# Run the example flow
if __name__ == "__main__":
    trello_api = TrelloAPI("nEnb2Pqa")
    print(trello_api.get_all_boards())
    trello_api.clear_board_and_lists()
    print(trello_api.get_all_card_details())
    # trello_api.create_custom_field("id_db", "text")
    # fields_data =trello_api.get_all_custom_fields()
    # print(json.dumps(fields_data, indent=2))
    #
    #
    # API_KEY = os.getenv('TRELLO_API_KEY')
    # TOKEN = os.getenv('TRELLO_API_TOKEN')
    # BOARD_ID = trello_api.board_id
    #
    # url = f"https://api.trello.com/1/boards/{BOARD_ID}/plugins?key={API_KEY}&token={TOKEN}"
    #
    # response = requests.get(url)
    #
    # if response.status_code == 200:
    #     plugins = response.json()
    #     for plugin in plugins:
    #         print(f"Plugin Name: {plugin.get('name')}, ID: {plugin.get('id')}")
    # else:
    #     print(f"❌ Failed to fetch plugins: {response.text}")
