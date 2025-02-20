import pandas as pd
import json
import numpy as np
import os
from openai import OpenAI

from globals import GLOBALS

class DBManager:
    def __init__(self):
        self.db_path = GLOBALS.db_path
        self.db_users_path = self.db_path + '/users.csv'
        self.db_tasks_path = self.db_path + '/tasks.csv'
        self.client = OpenAI(api_key=GLOBALS.openai_api_key)


    def create_db(self, rows_users, rows_tasks):
        if not os.path.exists(self.db_path):
            os.mkdir(self.db_path)
        pd.DataFrame(rows_users).to_csv(self.db_users_path, index=False)
        pd.DataFrame(rows_tasks).to_csv(self.db_tasks_path, index=False)


    def update_tasks(self, tasks):
        df = pd.read_csv(self.db_tasks_path)
        for task in tasks:
            if 'id' in task and task['id'] in df['id'].values:
                df.update(pd.DataFrame([task]))
            else:
                df = pd.concat([df, pd.DataFrame([task])], ignore_index=True)
        df.to_csv(self.db_tasks_path, index=False)

    def read_tasks(self):
        df = pd.read_csv(self.db_tasks_path)
        return df

    def update_users(self, users):
        df = pd.read_csv(self.db_users_path)
        for user in users:
            if 'id' in user and user['id'] in df['id'].values:
                df.update(pd.DataFrame([user]))
            else:
                df = pd.concat([df, pd.DataFrame([user])], ignore_index=True)
        df.to_csv(self.db_users_path, index=False)

    def writ_columns_to_db(self, type_db, columns):
        df = pd.read_csv(getattr(self, type_db))
        for col, default_value in columns.items():
            if col not in df.columns:
                df[col] = default_value
        df.to_csv(getattr(self, type_db), index=False)

    def load_user_embeddings(self):
        if not self.db_users_path or not os.path.exists(self.db_users_path):
            return {}
        df = pd.read_csv(self.db_users_path)
        embeddings = {}
        for _, row in df.iterrows():
            if row.get('embedding'):
                embeddings[row['hebrew_name_english']] = np.array(json.loads(row['embedding']))
        return embeddings

    def save_user_embeddings(self, embeddings):
        df = pd.read_csv(self.db_users_path)
        for user_id, emb in embeddings.items():
            df.loc[df['hebrew_name_english'] == user_id, 'embedding'] = json.dumps(emb.tolist())
        df.to_csv(self.db_users_path, index=False)

    def generate_updates_from_model(self, new_data):
        """
        Use a model to generate updates by comparing existing data from the DB with new data from classification.
        """
        existing_data = self.read_tasks()

        prompt = f"""Compare the existing project data below with the new extracted data from the meeting.
        Generate a list of updates with necessary changes, additions, or removals.

        Existing Data: {json.dumps(existing_data)}
        New Data: {json.dumps(new_data)}

        Return JSON list of updates, each with keys: id, field, new_value.
        """

        response = self.client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
        return json.loads(response.choices[0].message.content)