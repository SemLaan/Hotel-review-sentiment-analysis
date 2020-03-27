
from sqlalchemy import create_engine
import pymysql
import pandas as pd


class MysqlDB:

    def __init__(self):

        self.engine = create_engine('mysql+pymysql://pyconnect:POPdievuileAAP@localhost/hotel_reviews', pool_recycle=3600)
        self.connection = self.engine.connect()
        #"localhost","pyconnect","POPdievuileAAP","hotel_reviews"

    def to_database(self, data, table_name):

        data.to_sql(table_name, self.connection, if_exists='replace')

    def from_database(self, subset):

        return pd.read_sql("CALL get_subset(\'" + subset + "\')", self.connection)

