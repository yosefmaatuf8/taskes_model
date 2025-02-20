import pandas as pd
import json
import os
from openai import OpenAI

from globals import GLOBALS


class DBManager:
    def __init__(self):
        self.db_path = GLOBALS.db_path
        self.db_users_path = self.db_path + '/users.csv'
        self.db_tasks_path = self.db_path + '/tasks.csv'
        self.db_topics_status_path = self.db_path + '/topics_status.csv'
        self.db_full_data_path = self.db_path + '/full_data.csv'
        self.db_meetings_transcriptions_path = self.db_path + '/meetings_transcriptions.csv'
        self.client = OpenAI(api_key=GLOBALS.openai_api_key)

    def create_db(self, rows_users, rows_tasks, rows_topics, rows_full_data, rows_meetings):
        if not os.path.exists(self.db_path):
            os.mkdir(self.db_path)

        pd.DataFrame(rows_users).to_csv(self.db_users_path, index=False)
        pd.DataFrame(rows_tasks).to_csv(self.db_tasks_path, index=False)
        pd.DataFrame(rows_topics).to_csv(self.db_topics_status_path, index=False)
        pd.DataFrame(rows_full_data).to_csv(self.db_full_data_path, index=False)
        pd.DataFrame(rows_meetings).to_csv(self.db_meetings_transcriptions_path, index=False)

    def update_db(self, type_db, data, id_column):
        file_path = getattr(self, type_db)
        df = pd.read_csv(file_path)

        for entry in data:
            if id_column in entry and entry[id_column] in df[id_column].values:
                df.update(pd.DataFrame([entry]))
            else:
                df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

        df.to_csv(file_path, index=False)

    def read_db(self, type_db):
        file_path = getattr(self, type_db)
        return pd.read_csv(file_path)

    def generate_updates_from_model(self, new_data):
        existing_data = self.read_db("db_tasks_path")
        prompt = f"""Compare the existing project data below with the new extracted data from the meeting.
        Generate a list of updates with necessary changes, additions, or removals.

        Existing Data: {json.dumps(existing_data.to_dict())}
        New Data: {json.dumps(new_data)}

        Return JSON list of updates, each with keys: id, field, new_value.
        """
        response = self.client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
        return json.loads(response.choices[0].message.content)
