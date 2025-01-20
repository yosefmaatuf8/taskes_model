from trello import TrelloClient
from dotenv import load_dotenv
import os

class TrelloAPI:
    def __init__(self):
        # Load .env file
        load_dotenv()
        # Initialize Trello Client
        self.client = TrelloClient(
                    api_key=os.getenv('TRELLO_API_KEY'),
                    api_secret=os.getenv('TRELLO_API_SECRET'),
                    token=os.getenv('TRELLO_API_TOKEN')
                )
        # Replace these with actual IDs from your environment
        self.board_id = os.getenv('TRELLO_BOARD_ID')


    def get_user_id_by_username(self, username):
        """Fetches the user ID from a username."""
        board = self.client.get_board(self.board_id)
        members = board.get_members()
        for member in members:
            if member.username == username:
                return member.id
        raise ValueError(f"User '{username}' not found on board with ID '{self.board_id}'.")

    # Function to get list ID by list name
    def get_list_id_by_name(self, list_name):
        """Fetches the list ID from a list name."""
        board = self.client.get_board(self.board_id)
        lists = board.all_lists()
        for trello_list in lists:
            if trello_list.name == list_name:
                return trello_list.id
        raise ValueError(f"List '{list_name}' not found on board with ID '{self.board_id}'.")

    # Function to get a card by its name
    def get_card_by_name(self, card_name):
        """Fetches a card object by its name."""
        board = self.client.get_board(self.board_id)
        for trello_list in board.all_lists():
            for card in trello_list.list_cards():
                if card.name == card_name:
                    return card
        raise ValueError(f"Card '{card_name}' not found on board with ID '{self.board_id}'.")

    # Function to create a new card in a list
    def create_card(self,list_name, card_name, card_description):
        """Creates a new card in the specified list by name."""
        list_id = self.get_list_id_by_name(list_name)
        board = self.client.get_board(self.board_id)
        trello_list = board.get_list(list_id)
        card = trello_list.add_card(card_name, card_description)
        print(f"Card '{card_name}' created in list '{list_name}'.")
        return card

    # Function to assign a member to a card
    def assign_member_to_card(self, card_name, username):
        """Assigns a user to the specified card using their username."""
        card = self.get_card_by_name(card_name)
        user_id = self.get_user_id_by_username(username)
        card.assign(user_id)
        print(f"Assigned user '{username}' to card '{card_name}'.")

    # Function to move a card between lists
    def move_card_to_list(self,card_name, target_list_name):
        """Moves a card to the specified list by name."""
        card = self.get_card_by_name(card_name)
        target_list_id = self.get_list_id_by_name(target_list_name)
        card.change_list(target_list_id)
        print(f"Moved card '{card_name}' to list '{target_list_name}'.")

    # Function to comment on a card
    def comment_on_card(self,card_name, comment):
        """Adds a comment to the specified card by name."""
        card = self.get_card_by_name(card_name)
        card.comment(comment)
        print(f"Added comment on card '{card_name}': {comment}")


    # Function to get all card details
    def get_all_card_details(self):
        """Fetches all card details including users, description, and list information."""
        board = self.client.get_board(self.board_id)
        cards_details = {}

        for trello_list in board.all_lists():
            for card in trello_list.list_cards():
                # Fetch member usernames using 'idMembers'
                card_members = [self.client.get_member(member_id).username for member_id in card.member_ids]

                cards_details[card.name] = {
                    "users": card_members,
                    "description": card.desc,
                    "list": trello_list.name
                }

        return cards_details

# Run the example flow
if __name__ == "__main__":
    # comment_on_card('Creating action items', 'comment')
    print(TrelloAPI.get_all_card_details())

