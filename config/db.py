import os
import pandas as pd
from pathlib import Path
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

class Database:
    def __init__(self):
        """
        No init, carregamos as variáveis e configuramos a engine.
        A engine mantém um 'pool' de conexões, mas não conecta imediatamente.
        """
        env_path = Path(__file__).resolve().parent.parent / '.env'
        load_dotenv(env_path, override=True)

        params = quote_plus(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={os.getenv('DB_SERVER')};"
            f"DATABASE={os.getenv('DB_NAME')};"
            f"UID={os.getenv('DB_USER')};"
            f"PWD={os.getenv('DB_PASSWORD')}"
        )

        self.engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    def query(self, sql_query):
        """Executa SELECT e retorna DataFrame"""
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql_query), conn)

    def upload_df(self, df, table_name, if_exists='append', dtype=None):
        """Insere DataFrame no banco"""
        with self.engine.begin() as conn:
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists=if_exists,
                index=False,
                dtype=dtype
            )

    def execute_command(self, sql_cmd):
        """Executa comandos sem retorno (UPDATE, DELETE)"""
        with self.engine.begin() as conn:
            conn.execute(text(sql_cmd))


db = Database()