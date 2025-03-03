from trello_api import TrelloAPI
from db_manager import DBManager
from collections import defaultdict

class ExtractAndRunFunctions:
    def __init__(self, trello_api=TrelloAPI(), db_manager=DBManager()):
        self.trello_api = trello_api
        self.db_manager = db_manager

    def execute_tasks_directly(self, updates):
        """
        Processes DB updates and applies them directly to Trello.
        - Each list in Trello represents a topic.
        - The first card in each list represents the topic's status.
        - All other cards in the list are tasks, each having its own status.
        """
        updated_tasks = updates.get("updated_tasks", [])
        updated_statuses = updates.get("updated_statuses", [])

        # Fetch existing Trello tasks
        existing_trello_tasks = self.trello_api.get_all_card_details()
        existing_task_names = {name.lower(): details for name, details in existing_trello_tasks.items()}
        existing_lists = {details["list"].lower(): details["list"] for details in existing_trello_tasks.values()}

        # Fetch status data from DB
        df_statuses = self.db_manager.read_db("db_topics_status_path")
        topic_status_map = defaultdict(lambda: "Unknown")

        if not df_statuses.empty and "topic" in df_statuses.columns and "topic_status" in df_statuses.columns:
            topic_status_map.update({row["topic"]: row["topic_status"] for _, row in df_statuses.iterrows()})

        # Process updated tasks
        for task in updated_tasks:
            task_id = task.get("id", "").strip()
            topic = task.get("topic", "").strip()
            task_name = task.get("name", topic).strip()  # Ensure task_name is not empty
            new_status = task.get("status", "").strip()
            assigned_user = task.get("assigned_user", "").strip()

            existing_card = existing_task_names.get(task_name.lower())

            if existing_card:
                print(f"Updating existing card '{task_name}'")
                self.trello_api.update_card(
                    card_name=task_name,
                    task_id=task_id,
                    new_status=new_status,
                    new_assigned_user=assigned_user
                )

            else:
                # Ensure the list exists before creating a card
                if topic.lower() not in existing_lists:
                    print(f"Creating list '{topic}'")
                    self.trello_api.create_list(topic)

                # Create the first card in the list (Status of the Topic) if missing
                status_card_name = f"Status of {topic}"
                status_description = f"Current Status: {topic_status_map.get(topic, 'Unknown')}"
                existing_status_card = self.trello_api.get_card_by_name(status_card_name)

                if not existing_status_card:
                    print(f"Creating status card '{status_card_name}' in list '{topic}'")
                    self.trello_api.create_card(topic, status_card_name, status_description, "", "", "")

                # Create the new task card
                print(f"Creating new card '{task_name}' in list '{topic}'")
                new_card_id = self.trello_api.create_card(
                    topic,
                    task_name,
                    "",
                    task_id,
                    new_status,
                    assigned_user
                )
                # Assign ID, status, and user in description
                if new_card_id:
                    print(f"Assigning id_db '{task_id}', status '{new_status}', and user '{assigned_user}' to new card '{task_name}'")
                    self.trello_api.update_card(
                        card_name=task_name,
                        task_id=task_id,
                        new_status=new_status,
                        new_assigned_user=assigned_user
                    )

        # Update status cards in lists
        for topic, status in topic_status_map.items():
            list_id = self.trello_api.get_list_id_by_name(topic)
            if list_id:
                status_card_name = f"Status of {topic}"
                existing_status_card = self.trello_api.get_card_by_name(status_card_name)

                if existing_status_card:
                    print(f"Updating status card for '{topic}'")
                    self.trello_api.update_card(status_card_name, new_description=f"Current Status: {status}")
                else:
                    print(f"Creating missing status card for '{topic}'")
                    self.trello_api.create_card(topic, status_card_name, "","",f"Current Status: {status}","")
