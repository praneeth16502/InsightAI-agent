import re
import subprocess

class DataAgent:
    def __init__(self, model="llama3:latest"):
        self.model = model

    def _format_schema(self, metadata: dict) -> str:
        return "\n".join(
            f"{table}({', '.join(cols)})"
            for table, cols in metadata.items()
        )

    def _call_llm(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt,
            text=True,
            capture_output=True
        )
        return result.stdout.strip()

    def generate_sql_candidates(self, question, metadata, k=4):
        schema = self._format_schema(metadata)

        prompt = f"""
You are a SQLite expert.

Schema:
{schema}

Generate {k} DIFFERENT valid SQL queries for:
{question}

Rules:
- Use only schema tables/columns
- Each query on a new line
- No explanations
- End each query with semicolon
"""
        response = self._call_llm(prompt)
        return self._extract_all_sql(response)[:k]

    def select_best_sql(self, sql_candidates, db):
        best_sql = None
        best_score = -1

        for sql in sql_candidates:
            df, error = db.execute_query(sql)
            if error or df is None:
                continue

            score = 0
            score += len(df)                 # rows
            score += df.shape[1] * 5         # columns weighted higher

            if score > best_score:
                best_score = score
                best_sql = sql

        return best_sql

    def generate_sql(self, question, metadata, db):
        candidates = self.generate_sql_candidates(question, metadata)
        return self.select_best_sql(candidates, db)

    def _extract_all_sql(self, text):
        return re.findall(r"(SELECT .*?;)", text, re.S | re.I)
