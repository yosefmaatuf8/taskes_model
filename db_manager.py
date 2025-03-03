import pandas as pd
import logging
import uuid
import json
import numpy as np
import os
from openai import OpenAI
import tiktoken
from globals import GLOBALS
from utils.utils import extract_json_from_code_block, split_text


class DBManager:
    def __init__(self):
        self.updated_for_trello = None
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.updated_tasks_log = {}
        self.data_topics_and_tasks = None
        self.existing_topics = None
        self.trello_tasks = GLOBALS.list_tasks
        self.trello_data = None
        self.db_path = GLOBALS.db_path
        self.db_users_path = self.db_path + '/users.csv'
        self.db_tasks_path = self.db_path + '/tasks.csv'
        self.db_topics_status_path = self.db_path + '/topics_status.csv'
        self.db_full_data_path = self.db_path + '/full_data.csv'
        self.db_meetings_transcriptions_path = self.db_path + '/meetings_transcriptions.csv'
        self.client = OpenAI(api_key=GLOBALS.openai_api_key)
        self.users_data_str = None
        if os.path.exists(self.db_full_data_path):
            self.users_data_str = self.read_users_data()[1]

    def create_db(self, rows_users, rows_tasks, rows_topics, rows_full_data, rows_meetings):
        if not os.path.exists(self.db_path):
            os.mkdir(self.db_path)

        pd.DataFrame(rows_users).to_csv(self.db_users_path, index=False)
        pd.DataFrame(rows_tasks).to_csv(self.db_tasks_path, index=False)
        pd.DataFrame(rows_topics).to_csv(self.db_topics_status_path, index=False)
        pd.DataFrame(rows_full_data).to_csv(self.db_full_data_path, index=False)
        pd.DataFrame(rows_meetings).to_csv(self.db_meetings_transcriptions_path, index=False)

    def update_db(self, type_db, data, id_column):
        """
        Updates the specified database file with new or modified records.
        Ensures proper merging without overwriting unintended data.
        """
        file_path = getattr(self, type_db)

        # ✅ Load existing database if available, otherwise create an empty DataFrame
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            df = pd.DataFrame(columns=[id_column, "topic", "name", "status", "assigned_user", "summary"])

        existing_ids = set(df[id_column].dropna().astype(str))  # ✅ Get existing IDs as a set

        for entry in data:
            entry_id = entry.get(id_column)

            # ✅ If the ID exists, update the existing record
            if entry_id in existing_ids:
                idx = df[df[id_column] == entry_id].index[0]

                for key, value in entry.items():
                    if key in df.columns and pd.notna(value):
                        df.at[idx, key] = value

            # ✅ If the ID is missing or new, add a new record
            else:
                df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

        # ✅ Save the updated DataFrame back to CSV
        df.to_csv(file_path, index=False)

    def read_db(self, type_db):
        file_path = getattr(self, type_db)
        return pd.read_csv(file_path)

    def read_users_data(self):
        df = pd.read_csv(self.db_users_path)
        users_data = {}
        for _, row in df.iterrows():
            id_user, user_name, hebrew_name, full_name_english = row.get("id"), row.get("name"), row.get("hebrew_name"), row.get("full_name_english")
            users_data[full_name_english] = (id_user, user_name, hebrew_name)
        return users_data, str(users_data)

    def get_meeting_datetime(self, meeting_id: str):
        """Retrieve meeting datetime from CSV."""
        try:
            df = pd.read_csv(self.db_meetings_transcriptions_path)
            meeting_row = df[df["meeting_id"] == meeting_id]

            if meeting_row.empty:
                print(f"Meeting ID {meeting_id} not found.")
                return None

            meeting_datetime = meeting_row.iloc[0].get("meeting_datetime", "").strip()
            if not meeting_datetime:
                print(f"No datetime found for Meeting ID {meeting_id}.")
                return None

            return meeting_datetime

        except Exception as e:
            print(f"Error retrieving meeting datetime: {e}")
            return None

    def load_user_embeddings(self):
        embeddings = {}

        if not self.db_users_path or not os.path.exists(self.db_users_path):
            logging.warning("Database user file path is invalid or does not exist.")
            return embeddings

        try:
            df = pd.read_csv(self.db_users_path)
        except Exception as e:
            logging.error(f"Error loading CSV file: {e}")
            return embeddings

        for _, row in df.iterrows():
            embedding_str = row.get("embedding", "").strip()
            hebrew_name_eng = row.get("hebrew_name_english", "").strip()

            if not embedding_str or not hebrew_name_eng:
                logging.warning(f"Skipping row due to missing data: {row}")
                continue

            try:
                embeddings[hebrew_name_eng] = np.array(json.loads(embedding_str))
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Skipping invalid embedding for {hebrew_name_eng}: {e}")

        return embeddings


    def save_user_embeddings(self, embeddings):
        df = pd.read_csv(self.db_users_path)
        for user_id, emb in embeddings.items():
            df.loc[df['hebrew_name_english'] == user_id, 'embedding'] = json.dumps(emb.tolist())
        df.to_csv(self.db_users_path, index=False)

    def extract_topics_from_transcription(self, transcription_text):
        """
        Sends the transcription to the model and extracts structured topics, tasks, statuses,
        and a summarized transcription for each task.
        """
        prompt = f'''
        You are an AI assistant specializing in meeting analysis.
        You are **fully capable of processing Hebrew**.
        Do **not** translate the text. Extract the data as-is.

        Your task is to extract structured project-related information from the following meeting transcription.
        Identify the relevant project topics, associated tasks, assigned users, current statuses, 
        and provide a **concise summary** of what was said about each task.
        ---

        **❗ Guidelines:**
        1. **Only extract significant project-related topics and tasks.**  
           - **A task must require action, tracking, and clear responsibility.**  
           - **Ignore** casual statements or vague suggestions.
        2. **A task must include a clear action, assigned user, and expected status.**
           - Valid: "Prepare the API documentation by Thursday."  
           - Invalid: "We should discuss the API."
        3. **Tasks must be grouped under relevant topics** (use only the existing topics unless a new topic is clearly required).  
        4. **Assigned users should only be selected from the provided list of users and usernames.
           - If a person's name appears in the transcription, try to match it with the most appropriate username from the provided list.
           - If the exact name is not found, select the closest match based on context.
           - If no clear user is mentioned, or if the name cannot be confidently matched, set "assigned_user": "unknown".
        5. **Every task must have an assigned user and a status.**
           - If the status is not explicitly mentioned, infer it logically based on the conversation.
           - If no user is mentioned, assign `"assigned_user": "unknown"`.

        ---
        **Meeting Transcription (in Hebrew, do not translate):**
        {transcription_text}

        ---
        **Existing Topics (for reference):**
        {self.existing_topics}

        ---
        **Expected JSON Output Format:**
        ```json
        {{
          "topics": [
            {{
              "name": "Topic Name (must be written in English only)",
              "tasks": [
                {{
                  "id": "Task ID",
                  "name": "Task Name (must be written in English only)",
                  "status": "Task Status (To Do, In Progress, Completed)",
                  "assigned_user": "Assigned User (must be from the provided list or 'unknown')",
                  "previous_status": "Previous Task Status (if applicable)",
                  "summary": "Concise summary of what was said about this task during the meeting (must be written in English only)"
                }}
              ]
            }}
          ]
        }}
        ```
        **Return ONLY valid JSON format. No explanations, introductions, or additional text.**
        '''

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a structured data extraction assistant capable of processing Hebrew."},
                {"role": "system", "content": f"This is the list of users and their usernames: {self.users_data_str}"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.1
        )
        return extract_json_from_code_block(response.choices[0].message.content)

    def generate_updates_from_model(self, topic_name, new_data, type_db, trello_tasks="not_relevant"):
        """
        Generates updates for a **single topic at a time** by comparing it with existing data.
        Ensures task names are translated to English and returns only fields that exist in the new data.
        """

        # Read existing data
        existing_topic_data = self.read_db(type_db)
        if not existing_topic_data.empty and "topic" in existing_topic_data.columns:
            existing_topic_data = existing_topic_data[existing_topic_data["topic"] == topic_name].to_dict(
                orient="records")
        else:
            existing_topic_data = []

        # Ensure all tasks have a unique ID
        for task in (new_data if isinstance(new_data, list) else new_data.get("tasks", [])):
            if not task.get("id") or task["id"] in self.users_data_str:
                task["id"] = str(uuid.uuid4())[:5]

        # Serialize JSON safely
        json_existing_data = json.dumps(existing_topic_data, ensure_ascii=False, indent=2)
        json_new_data = json.dumps(new_data, ensure_ascii=False, indent=2)
        json_trello_tasks = json.dumps(trello_tasks, ensure_ascii=False,
                                       indent=2) if trello_tasks != "not_relevant" else "N/A"

        # Construct the prompt
        prompt = f"""
        You are an AI assistant responsible for managing tasks in project meetings.

        ## Instructions:
        1. **Translate all task names to English** while keeping the original meaning.
        2. **Match tasks by name when comparing to Trello**.
        3. **Update existing tasks** in Trello instead of creating duplicates.
        4. **If a task is new**, add it with a new ID.
        5. **Do NOT extract questions or general statements as tasks.**
        6. **Ensure clean and structured JSON output.**
        7. **If a field is missing in new data, do NOT return that field at all.**

        ---

        **Processing Topic:** "{topic_name}"

        **Existing Data for this Topic:**
        ```json
        {json_existing_data}
        ```

        **New Data for this Topic:**
        ```json
        {json_new_data}
        ```

        **Relevant Trello Tasks:**
        ```json
        {json_trello_tasks}
        ```

        ---
        **Expected JSON Output Format:**
        ```json
        [
          {{
            "id": "task_unique_id",
            "name": "Updated Task Name",
            "status": "Updated Status",
            "assigned_user": "Updated Assigned User",
            "summary": "Updated Summary",

          }}
        ]
        ```
        """

        # Send the prompt to GPT
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.2
        )

        # Extract structured JSON output
        updates = extract_json_from_code_block(response.choices[0].message.content)

        if not updates:
            print(f"No updates returned for topic {topic_name}")
            return []

        return updates

    def update_full_data(self, data_topic_and_tasks, topic_name):
        """
        Updates full_data.csv with the latest transcription results.
        Ensures all fields are correctly populated and propagates changes to tasks.csv.
        """
        df_existing = self.read_db("db_full_data_path")
        df_for_update = self.generate_updates_from_model(topic_name, data_topic_and_tasks, "db_full_data_path",
                                                         self.trello_tasks)


        existing_ids = set(df_existing["id"].dropna().astype(str))  # ✅ Get existing IDs
        updated_tasks_log = []

        for update in df_for_update:
            task_id = update.get("id", "").strip()

            # ✅ Generate a unique ID if the task doesn't exist
            if not task_id or task_id not in existing_ids:
                new_id = str(uuid.uuid4())[:5]
                task_id = new_id
                update["id"] = task_id

                # ✅ Extract previous values if the task exists
            existing_entry = df_existing[df_existing["id"] == task_id]
            if not existing_entry.empty:
                before_update_value = existing_entry.iloc[0].to_dict()
            else:
                before_update_value = {}  # New task, no previous data

            update_dict = {}

            for field, new_value in update.items():
                if field == "id":
                    continue  # Skip ID since it's already processed

                old_value = before_update_value.get(field, "")

                if str(new_value) != str(old_value):  # ✅ Update only if there's a real change
                    if task_id in df_existing["id"].values:
                        df_existing.loc[df_existing["id"] == task_id, field] = new_value
                    else:
                        new_task_entry = {
                            "id": task_id,
                            "topic": topic_name,
                            "name": update.get("name", ""),
                            "status": update.get("status", ""),
                            "assigned_user": update.get("assigned_user", ""),
                            "summary": update.get("summary", "")
                        }
                        df_existing = pd.concat([df_existing, pd.DataFrame([new_task_entry])], ignore_index=True)

                    update_dict[field] = new_value

            if update_dict:  # ✅ Add only if changes were made
                updated_tasks_log.append({
                    "id": task_id,
                    "before_update": before_update_value,
                    "update": update_dict
                })

        df_existing.to_csv(self.db_full_data_path, index=False)

        # ✅ Update DB and propagate changes properly
        self.update_tasks_from_full_data(topic_name, updated_tasks_log)

        # ✅ Correctly store `self.updated_tasks_log`
        if topic_name in self.updated_tasks_log:
            self.updated_tasks_log[topic_name].extend(updated_tasks_log)
        else:
            self.updated_tasks_log[topic_name] = updated_tasks_log

    def update_tasks_from_full_data(self, topic_name, updated_tasks_log):
        """
        Updates the tasks database based on changes made in any source.
        Ensures that only predefined fields are updated and no new columns are added.
        """
        df_tasks = self.read_db("db_tasks_path")
        allowed_fields = {"id", "topic", "name", "status", "assigned_user"}  # Allowed fields for update

        existing_ids = set(df_tasks["id"].dropna().astype(str))  # Get existing IDs as a set

        for update in updated_tasks_log:
            task_id = update.get("id")
            update_data = update.get("update", {})

            if not task_id:
                continue

            # ✅ If task exists, update relevant fields
            if task_id in existing_ids:
                for field, new_value in update_data.items():
                    if field in allowed_fields:
                        df_tasks.loc[df_tasks["id"] == task_id, field] = new_value

            # ✅ If task is new, add it with only the allowed fields
            else:
                new_task_entry = {
                    "id": task_id,
                    "topic": topic_name,  # Ensure topic is set
                    "name": update_data.get("name", ""),
                    "status": update_data.get("status", ""),
                    "assigned_user": update_data.get("assigned_user", "")
                }
                df_tasks = pd.concat([df_tasks, pd.DataFrame([new_task_entry])], ignore_index=True)

        # ✅ Save the updated DataFrame back to CSV
        df_tasks.to_csv(self.db_tasks_path, index=False)

    def update_status(self, topic_name):
        """
        Updates the db_topics_status database for a specific topic.
        Ensures that all task IDs related to the topic are correctly stored.
        """
        tasks_data = self.read_db("db_tasks_path")

        if topic_name in tasks_data["topic"].values:
            # Filter only relevant tasks
            topic_tasks = tasks_data[tasks_data["topic"] == topic_name]

            # ✅ Get the most common status (mode), fallback to "Unknown" if empty
            topic_status = topic_tasks["status"].mode()
            topic_status = topic_status.iloc[0] if not topic_status.empty else "Unknown"

            # ✅ Collect all task IDs related to the topic
            task_ids = topic_tasks["id"].dropna().astype(str).tolist()

            # ✅ Ensure unique task IDs and sort them (for consistency)
            task_ids = sorted(set(task_ids))

            # ✅ Prepare the update dictionary
            status_update = [{
                "topic": topic_name,
                "topic_status": topic_status,
                "tasks_id": ",".join(task_ids)  # Store task IDs as a comma-separated string
            }]

            # ✅ Update the database
            self.update_db("db_topics_status_path", status_update, "topic")

    @staticmethod
    def merge_topics(existing_data, new_data):
        """
        Merges topic data from different chunks.
        - If a topic appears multiple times, its tasks will be merged with existing tasks.
        - If a task appears multiple times, its status will be updated accordingly.

        :param existing_data: The existing topic data.
        :param new_data: The new data extracted from the current chunk.
        :return: A unified structure containing merged topics and tasks.
        """
        topic_dict = {topic["name"]: topic for topic in existing_data.get("topics", [])}
        if not isinstance(new_data, dict):
            print("new_data is not dict", new_data)
            return []
        for topic in new_data.get("topics", []):
            topic_name = topic["name"]

            if topic_name in topic_dict:
                # If the topic already exists, add new tasks or update existing ones.
                existing_tasks = {task["id"]: task for task in topic_dict[topic_name]["tasks"]}

                for task in topic["tasks"]:
                    task_id = task["id"]
                    if task_id in existing_tasks:
                        # If the task already exists, update it only if the status has changed.
                        if task["status"] != existing_tasks[task_id]["status"]:
                            existing_tasks[task_id]["status"] = task["status"]
                        if task.get("assigned_user") and task["assigned_user"] != existing_tasks[task_id][
                            "assigned_user"]:
                            existing_tasks[task_id]["assigned_user"] += task["assigned_user"]
                        if task.get("summary") and existing_tasks[task_id].get("summary"):
                            existing_tasks[task_id]["summary"] += task["summary"]
                    else:
                        # If this is a new task, add it.
                        existing_tasks[task_id] = task

                # Update the task list for the topic.
                topic_dict[topic_name]["tasks"] = list(existing_tasks.values())
            else:
                # If the topic is entirely new, add it.
                topic_dict[topic_name] = topic

        return {"topics": list(topic_dict.values())}

    def get_updated_tasks(self):
        """
        Extracts only tasks that have been updated in the current session.
        Returns a list of changed tasks in the required format for Trello updates.
        """
        updated_tasks = {}

        # Iterate over the logged task updates
        for topic_name, updates in self.updated_tasks_log.items():
            for update in updates:
                task_id = update.get("id")
                new_update = update.get("update", {})

                if not task_id or not new_update:
                    continue

                # If task already exists, update its fields
                if task_id in updated_tasks:
                    updated_tasks[task_id].update(new_update)
                else:
                    # Create new entry with only the updated fields
                    updated_tasks[task_id] = {
                        "id": task_id,
                        "topic": topic_name
                    }
                    updated_tasks[task_id].update(new_update)  # Add only new values

        return list(updated_tasks.values())  # Convert dictionary to list

    def get_updated_statuses(self):
        """
        Extracts only topic statuses that have been updated in the current session.
        Returns a list of changed topic statuses for Trello updates.
        """
        updated_statuses = []

        df_statuses = self.read_db("db_topics_status_path")

        for _, row in df_statuses.iterrows():
            topic_name = row.get("topic")
            if topic_name in self.updated_tasks_log:
                status_update = {
                    "topic": topic_name,
                    "topic_status": row.get("topic_status", ""),
                    "tasks_id": row.get("tasks_id", "")
                }
                updated_statuses.append(status_update)

        return updated_statuses

    def get_trello_updates(self):
        """
        Combines only updated tasks and statuses into a single JSON structure.
        This will be used as input for Trello updates.
        """
        updated_tasks = self.get_updated_tasks()
        updated_statuses = self.get_updated_statuses()

        trello_updates = {
            "updated_tasks": updated_tasks if updated_tasks else [],
            "updated_statuses": updated_statuses if updated_statuses else []
        }

        return trello_updates

    def update_project_data(self, processed_transcription, meeting_id, meeting_datetime):
        """
        Updates project data by merging repeated topics from different chunks.
        """
        if not self.users_data_str:
            self.users_data_str = self.users_data_str = self.read_users_data()[1]

        # 1) Read existing topics.
        df_topics = self.read_db("db_topics_status_path")
        self.existing_topics = df_topics["topic"].unique().tolist() if not df_topics.empty else []

        # 2) Convert transcription to JSON.
        refined_list = processed_transcription.get("refined_transcription", [])
        transcription_str = json.dumps({"refined_transcription": refined_list})

        # 3) Split into chunks if needed.
        max_length = 2000
        combined_data = {"topics": []}

        if len(transcription_str) > max_length:
            text_chunks = split_text(self.tokenizer, transcription_str, max_length)
            for chunk_text in text_chunks:
                chunk_result = self.extract_topics_from_transcription(chunk_text)
                combined_data = self.merge_topics(combined_data, chunk_result)
        else:
            combined_data = self.extract_topics_from_transcription(transcription_str)

        self.data_topics_and_tasks = combined_data  # Store merged topic data.

        if not self.data_topics_and_tasks or "topics" not in self.data_topics_and_tasks:
            print("No topics found; nothing to update.")
            return

        # 4) Process each topic separately.
        for topic_info in self.data_topics_and_tasks["topics"]:
            topic_name = topic_info["name"]
            tasks = topic_info["tasks"]

            self.update_full_data(tasks, topic_name)
            self.update_status(topic_name)

        # 5) Synchronize with Trello.
        self.updated_for_trello = self.get_trello_updates()

        # 6) Save meeting transcription data
        topics = [topic["name"] for topic in self.data_topics_and_tasks.get("topics", [])]
        transcription_json = json.dumps(processed_transcription, ensure_ascii=False)  # Convert to JSON string

        rows_meetings = [
            {
                "meeting_id": meeting_id,
                "transcription": transcription_json,  # Store transcription as JSON string
                "topics": json.dumps(topics, ensure_ascii=False), # Store topics as JSON string
                "meeting_datetime": meeting_datetime
            }
        ]

        # Ensure `meeting_id` is correctly used as the ID column
        self.update_db("db_meetings_transcriptions_path", rows_meetings, "meeting_id")
        output =  {
            "data_topics_and_tasks": self.data_topics_and_tasks,
            "updated_topics": self.updated_tasks_log,
            "updated_for_trello": self.updated_for_trello
        }
        return output
