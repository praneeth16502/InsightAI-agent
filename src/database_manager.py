import sqlite3


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_metadata(self):
        """Return schema: {table: [columns]}"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]

        metadata = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [row[1] for row in cursor.fetchall()]
            metadata[table] = columns

        conn.close()
        return metadata

    def execute_query(self, sql: str):
        """Execute SQL safely."""
        if not sql:
            return None, "Empty SQL"

        try:
            conn = sqlite3.connect(self.db_path)
            df = None
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            conn.close()

            import pandas as pd
            df = pd.DataFrame(rows, columns=columns)
            return df, None
        except Exception as e:
            return None, str(e)
