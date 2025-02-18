import pandas as pd
import json
import numpy as np
import os

from tensorflow.python.keras.combinations import generate

from globals import GLOBALS
class DBManager:
    def __init__(self):
        self.db_tasks_path = None
        self.db_users_path = None
        self.db_path = GLOBALS.db_path

    def create_db(self,rows_users,rows_tasks):
        if not os.path.exists(self.db_path):
            os.mkdir(self.db_path)
        df_users = pd.DataFrame(rows_users)
        df_tasks = pd.DataFrame(rows_tasks)
        self.db_users_path = self.db_path+'/users.csv'
        self.db_tasks_path = self.db_path+'/tasks.csv'
        df_users.to_csv(self.db_users_path,index=False)
        df_tasks.to_csv( self.db_tasks_path, index=False)

    def writ_rows_to_db(self,type_db,rows):
        df = pd.read_csv(getattr(self, type_db))
        new_rows = pd.DataFrame(rows)
        df = pd.concat([df,new_rows],ignore_index = True)
        df.to_csv(getattr(self, type_db), index=False)


    def writ_columns_to_db(self, type_db, columns):
        df = pd.read_csv(getattr(self, type_db))  # Read existing data

        for col, default_value in columns.items():
            if col not in df.columns:  # Only add new columns
                df[col] = default_value

        df.to_csv(getattr(self, type_db), index=False)  # Save back to file

    def read_from_db(self,type_db,id_row):
        pass


    def read_all_db(self):
        pass

    def load_user_embeddings(self):
        """
        Reads users.csv, parses the 'embedding' column (JSON string) into a dict:
           {row_identifier: np.array([...])}
        row_identifier could be the user 'id' or 'username' or something else
        depending on your data.
        """
        if not self.db_users_path or not os.path.exists(self.db_users_path):
            return {}

        df_users = pd.read_csv(self.db_users_path)
        embeddings_dict = {}

        # Decide how you'll identify each row. Could be 'id', 'username', or a dedicated 'speaker_label'
        identifier_col = 'id'  # or 'username' or 'speaker_label'

        for idx, row in df_users.iterrows():
            user_id = str(row[identifier_col])  # e.g. 'speaker_0' or '1234'
            emb_str = row.get('embedding', '')
            if emb_str:
                try:
                    # parse JSON string
                    vector_list = json.loads(emb_str)
                    emb_array = np.array(vector_list, dtype=float)
                    embeddings_dict[user_id] = emb_array
                except:
                    # if parsing fails, skip
                    pass

        return embeddings_dict

    def save_user_embeddings(self, embeddings_dict):
        """
        Writes the embeddings from embeddings_dict back into the 'embedding' column of users.csv.
        embeddings_dict: {row_identifier: np.array([...])}
        """
        if not self.db_users_path or not os.path.exists(self.db_users_path):
            print("Users CSV does not exist yet.")
            return

        df_users = pd.read_csv(self.db_users_path)

        identifier_col = 'hebrew_name_english'  # or 'username' or 'speaker_label', whichever you use

        # For each embedding, update the corresponding row in df_users
        for user_id, emb_array in embeddings_dict.items():
            # find the row(s) where df_users[identifier_col] == user_id
            mask = (df_users[identifier_col].astype(str) == str(user_id))
            if mask.any():
                # Convert np.array to JSON string
                emb_str = json.dumps(emb_array.tolist())
                df_users.loc[mask, 'embedding'] = emb_str
            else:
                # Possibly create a new row if not found
                # Depends on your logic
                new_row = {
                    identifier_col: user_id,
                    'embedding': json.dumps(emb_array.tolist())
                }
                df_users = df_users.append(new_row, ignore_index=True)

        df_users.to_csv(self.db_users_path, index=False)