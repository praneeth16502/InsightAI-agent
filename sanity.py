from src.database_manager import DatabaseManager
from src.agent_engine import DataAgent

db = DatabaseManager("data/chinook_sqlite.sqlite")
agent = DataAgent(model="llama3:latest")

metadata = db.get_metadata()

print("TABLES:", list(metadata.keys()))
print()

question = "Show all artists."
print("QUESTION:")
print(question)

sql_candidates = agent.generate_sql_candidates(question, metadata)

print("\nSQL CANDIDATES:")
for sql in sql_candidates:
    print(sql)

print("\nEXECUTION RESULTS:")
for sql in sql_candidates:
    df, error = db.execute_query(sql)
    print("\nSQL:", sql)
    print("ERROR:", error)
    if df is not None:
        print(df.head())
        break
