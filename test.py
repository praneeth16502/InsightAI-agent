import time
import pandas as pd
from src.database_manager import DatabaseManager
from src.agent_engine import DataAgent


db = DatabaseManager("data/chinook_sqlite.sqlite")
agent = DataAgent(model="llama3:latest")


def run_benchmark(benchmark_set):
    results = []
    metadata = db.get_metadata()

    print(f"\nStarting Benchmark: {len(benchmark_set)} questions\n")

    for test in benchmark_set:
        question = test["question"]
        gold_sql = test["gold_sql"]

        print("\n--------------------------------------------------")
        print("QUESTION:", question)

        start = time.time()

        # gold truth
        gold_df, _ = db.execute_query(gold_sql)

        # agent SQL
        sql = agent.generate_sql(question, metadata, db)
        print("SQL:", sql)

        pred_df, error = db.execute_query(sql)

        accuracy = False
        valid = error is None

        if valid and pred_df is not None:
            try:
                accuracy = pred_df.equals(gold_df)
            except Exception:
                accuracy = False

        latency = round(time.time() - start, 3)

        results.append({
            "Question": question,
            "Accuracy": accuracy,
            "Validity": valid,
            "Latency": latency
        })

    df = pd.DataFrame(results)

    print("\n=======================================================")
    print("FINAL BENCHMARK REPORT")
    print("=======================================================")
    summary = df.agg({
        "Accuracy": "mean",
        "Validity": "mean",
        "Latency": "mean"
    }).to_frame().T

    summary["Accuracy"] = (summary["Accuracy"] * 100).round(1).astype(str) + "%"
    summary["Validity"] = (summary["Validity"] * 100).round(1).astype(str) + "%"

    print(summary)
    print("=======================================================\n")

    return df


benchmark_set = [
    # very safe queries
    {"question": "Show all artists.", "gold_sql": "SELECT * FROM Artist;"},
    {"question": "List all albums.", "gold_sql": "SELECT * FROM Album;"},
    {"question": "Count all customers.", "gold_sql": "SELECT COUNT(*) FROM Customer;"},
    {"question": "Show all genres.", "gold_sql": "SELECT * FROM Genre;"},
    {"question": "Show all tracks.", "gold_sql": "SELECT * FROM Track LIMIT 50;"},

    # simple filters
    {"question": "Show customers from Brazil.", "gold_sql": "SELECT * FROM Customer WHERE Country='Brazil';"},
    {"question": "List invoices greater than 10.", "gold_sql": "SELECT * FROM Invoice WHERE Total > 10;"},
    {"question": "Show tracks longer than 3 minutes.", "gold_sql": "SELECT * FROM Track WHERE Milliseconds > 180000;"},

    # simple join
    {"question": "List album titles with artist names.",
     "gold_sql": "SELECT al.Title, ar.Name FROM Album al JOIN Artist ar ON al.ArtistId = ar.ArtistId;"}
]


if __name__ == "__main__":
    run_benchmark(benchmark_set)
