import pandas as pd
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