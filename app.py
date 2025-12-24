import streamlit as st
import plotly.express as px

# IMPORTANT: add src to Python path
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.database_manager import DatabaseManager
from src.agent_engine import DataAgent


st.set_page_config(
    page_title="InsightAgent | Autonomous Data Analytics",
    page_icon="🚀",
    layout="wide"
)

# DB path — now correctly points inside /data
db = DatabaseManager(db_path="data/chinook_sqlite.sqlite")
agent = DataAgent(model="llama3:latest")

# Custom CSS
st.markdown("""
<style>
.main { background-color: #f5f7f9; }
.stTextInput > div > div > input { background-color: #ffffff; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 InsightAgent: Autonomous Data Analytics")

st.markdown("""
This agent converts natural-language questions into SQL, 
executes them, and self-corrects when errors occur.
""")

# Sidebar
with st.sidebar:
    st.header("📊 Data Context")
    if st.button("View Database Schema"):
        st.text(db.get_metadata())
    st.info("Using local Llama-3 via Ollama (no external API calls).")

# Main input
query = st.text_input(
    "Ask a question about your data:",
    placeholder="e.g., Who are the top 5 customers by total spend?"
)

if query:
    metadata = db.get_metadata()

    with st.spinner("🤖 Agent is thinking..."):

        sql = agent.generate_sql(query, metadata,db=db)

        if sql:
            st.subheader("🛠️ Generated SQL")
            st.code(sql, language="sql")

            results, error = db.execute_query(sql)

            # Self-correction loop
            if error:
                st.error(f"Initial attempt failed: {error}")
                st.warning("🔄 Agent is analyzing the error and retrying...")

                fixed_sql = agent.generate_sql(
                    query, metadata, db=db, error_history=error
                )

                st.subheader("🛠️ Corrected SQL")
                st.code(fixed_sql, language="sql")

                results, error = db.execute_query(fixed_sql)

            # Output
            if results is not None and not results.empty:
                st.success("✅ Analysis Complete")

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Table View")
                    st.dataframe(results, use_container_width=True)

                with col2:
                    st.subheader("Visual Analysis")

                    if len(results.columns) >= 2:
                        x = results.columns[0]
                        y = results.columns[1]

                        try:
                            fig = px.bar(
                                results,
                                x=x,
                                y=y,
                                title=f"{y} by {x}",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            st.info("Could not visualize this result.")
                    else:
                        st.info("Single column — no chart generated.")

            elif results is not None and results.empty:
                st.info("Query executed successfully — but returned no data.")
            else:
                st.error(f"Agent could not resolve the error: {error}")

        else:
            st.error("Failed to generate SQL — is Ollama running?")

st.divider()
st.caption("InsightAgent | Built with Llama-3, Ollama, FAISS & Streamlit")
