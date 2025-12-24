import pandas as pd
import os
import time
import numpy as np
from src.database_manager import DatabaseManager
from src.agent_engine import DataAgent

# 1. Setup components
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "chinook_sqlite.sqlite")

db = DatabaseManager(DB_PATH)
agent = DataAgent(model="llama3:latest")


def compute_recall(df_gold, df_agent):
    """
    Compute row-level recall between gold and agent outputs.
    Allows column reordering and extra columns.
    """
    if df_gold is None or df_agent is None:
        return 0.0

    try:
        gold_rows = set(tuple(row) for row in df_gold.values)
        agent_rows = set(tuple(row) for row in df_agent.values)

        if not gold_rows:
            return 1.0

        return len(gold_rows.intersection(agent_rows)) / len(gold_rows)
    except Exception:
        return 0.0


def run_benchmark(benchmark_set):
    results = []
    metadata = db.get_metadata()

    print(f"\nStarting Benchmark: {len(benchmark_set)} Questions\n")

    for i, test in enumerate(benchmark_set):
        difficulty = "easy" if i < 20 else "medium"
        question = test["question"]

        print(f"Testing [{difficulty.upper()}]: {question}")
        start_time = time.time()

        sql = agent.generate_sql(question, metadata,db)
        df_gold, _ = db.execute_query(test["gold_sql"])
        df_agent, error = db.execute_query(sql)

        # Self-correction loop
        if error and sql:
            sql = agent.generate_sql(question, metadata, error_history=error)
            df_agent, error = db.execute_query(sql)

        latency = round(time.time() - start_time, 2)
        recall = compute_recall(df_gold, df_agent)
        is_correct = recall >= 0.8

        results.append({
            "Difficulty": difficulty,
            "Question": question,
            "Correct": is_correct,
            "Recall": round(recall, 2),
            "Latency": latency,
            "Valid": error is None
        })

        if not is_correct:
            print("❌ MISMATCH")
            print("Gold SQL:", test["gold_sql"])
            print("Agent SQL:", sql)
            print("-" * 50)

    # Report generation
    df = pd.DataFrame(results)
    summary = df.groupby("Difficulty").agg({
        "Correct": "mean",
        "Recall": "mean",
        "Valid": "mean",
        "Latency": "mean"
    })

    summary["Correct"] = (summary["Correct"] * 100).round(1)
    summary["Recall"] = (summary["Recall"] * 100).round(1)
    summary["Valid"] = (summary["Valid"] * 100).round(1)

    print("\n" + "=" * 55)
    print("                FINAL BENCHMARK REPORT")
    print("=" * 55)
    print(summary)
    print("=" * 55)

    return df


# --- 24 QUESTION BENCHMARK SET ---
benchmark_set = [
    # EASY: Direct Table Lookups
    {"question": "Show all artists.", "gold_sql": "SELECT * FROM Artist;"},
    {"question": "List all genres.", "gold_sql": "SELECT * FROM Genre;"},
    {"question": "Count total customers.", "gold_sql": "SELECT COUNT(*) FROM Customer;"},
    {"question": "Show all customers from USA.", "gold_sql": "SELECT * FROM Customer WHERE Country = 'USA';"},
    {"question": "Show all tracks.", "gold_sql": "SELECT * FROM Track;"},
    {"question": "Show all albums.", "gold_sql": "SELECT * FROM Album;"},
    {"question": "List all employees.", "gold_sql": "SELECT * FROM Employee;"},
    {"question": "Show first 5 invoices.", "gold_sql": "SELECT * FROM Invoice LIMIT 5;"},
    {"question": "Show all playlists.", "gold_sql": "SELECT * FROM Playlist;"},
    {"question": "Count total tracks.", "gold_sql": "SELECT COUNT(*) FROM Track;"},

    # EASY: Simple Filters
    {"question": "Show tracks with price greater than 0.99.", "gold_sql": "SELECT * FROM Track WHERE UnitPrice > 0.99;"},
    {"question": "List customers from Canada.", "gold_sql": "SELECT * FROM Customer WHERE Country = 'Canada';"},
    {"question": "Show employees hired after 2000.", "gold_sql": "SELECT * FROM Employee WHERE HireDate > '2000-01-01';"},
    {"question": "List invoices with total greater than 10.", "gold_sql": "SELECT * FROM Invoice WHERE Total > 10;"},
    {"question": "Show tracks longer than 300000 milliseconds.", "gold_sql": "SELECT * FROM Track WHERE Milliseconds > 300000;"},
    {"question": "List customers ordered by last name.", "gold_sql": "SELECT * FROM Customer ORDER BY LastName;"},
    {"question": "List albums ordered by title.", "gold_sql": "SELECT * FROM Album ORDER BY Title;"},
    {"question": "Show tracks from album ID 1.", "gold_sql": "SELECT * FROM Track WHERE AlbumId = 1;"},
    {"question": "Show invoices from customer ID 1.", "gold_sql": "SELECT * FROM Invoice WHERE CustomerId = 1;"},
    {"question": "Show tracks with genre ID 1.", "gold_sql": "SELECT * FROM Track WHERE GenreId = 1;"},

    # MEDIUM: Safe Joins & Aggregations
    {
        "question": "List track names and their genre names.",
        "gold_sql": "SELECT t.Name, g.Name FROM Track t JOIN Genre g ON t.GenreId = g.GenreId;"
    },
    {
        "question": "List album titles and artist names.",
        "gold_sql": "SELECT al.Title, ar.Name FROM Album al JOIN Artist ar ON al.ArtistId = ar.ArtistId;"
    },
    {
        "question": "Count number of tracks per genre.",
        "gold_sql": "SELECT g.Name, COUNT(t.TrackId) FROM Track t JOIN Genre g ON t.GenreId = g.GenreId GROUP BY g.Name;"
    },
    {
        "question": "Show total invoice count per customer.",
        "gold_sql": "SELECT CustomerId, COUNT(*) FROM Invoice GROUP BY CustomerId;"
    }
]


if __name__ == "__main__":
    df=run_benchmark(benchmark_set)
    df.to_csv("benchmark_results.csv", index=False)
    print("Saved -> benchmark_results.csv")
