import torch
import streamlit as st
from typing import Dict
from util.retrieve import retrieve
from util.generation import generate_sql
from util.init import init_models, initialize_graph
from util.self_improve import find_most_common_query_result
from util.load_data import load_schema, load_data_from_file, compute_embedding

def rag_query(db_id: str, query:Dict, k: int = 3) -> Dict:
    """
    One-liner for the full RAG pipeline:
      1. retrieve k neighbours,
      2. sample SQLs,
      3. pick the consensus execution.
    Returns a dict with the SQL and the examples retrieved.
    """

    retrieved = retrieve(query["embedding"], k=k)
    candidates = generate_sql(query, retrieved)
    db_path = f"spider_data/database/{db_id}/{db_id}.sqlite"
    best_sql = find_most_common_query_result(candidates, db_path)
    return {
        "question"          : query["question"],
        "generated_sql"     : best_sql,
    }

st.set_page_config(
    page_title="Spider RAG SQL Generator",
    page_icon="🕷️",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.query_history = []
    st.session_state.current_result = None

# Global variables
EMBEDDING_MODEL_NAME = "BAAI-bge-m3"
LLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize globals
if not st.session_state.initialized:
    
    schema, tables, columns_orig, foreign_pairs, pk_set = load_schema(
        schema_file="data/spider_schema.json",
        db_id="department_store"
    )

    query_embeds, documents, table_embeds, column_embeds = load_data_from_file()
    
    tokenizer, llm_model, embed_model = init_models(
        llm_model_name=LLM_MODEL_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        device=DEVICE
    )

    G = initialize_graph(
        tables=tables,
        columns_orig=columns_orig,
        foreign_pairs=foreign_pairs,
        pk_set=pk_set,
    ) 
    # Store in session state
    st.session_state.schema = schema
    st.session_state.tables = tables
    st.session_state.columns_orig = columns_orig
    st.session_state.foreign_pairs = foreign_pairs
    st.session_state.pk_set = pk_set
    st.session_state.G = G

# Main UI
st.title("🕷️ Spider RAG SQL Generator")
st.markdown("""
This interface generates SQL queries for the department store database using RAG (Retrieval-Augmented Generation).
Ask natural language questions and get SQL queries!
""")

# Main interface
st.header("💬 Ask a Question")

# Query input
user_question = st.text_input(
    "Enter your question about the department store database:",
    placeholder="e.g., What is the total sales amount for each department?"
)

# Example questions
with st.expander("📝 Example Questions"):
    st.write("Click on any example to use it:")
    example_questions = [
        "What is the total sales amount for each department?",
        "Which customers have made purchases over $100?",
        "List all products with their current stock levels",
        "Show the top 5 best-selling products",
        "Find all orders placed in the last month",
        "What are the names of all departments?",
        "Show me all employees and their department names"
    ]
    cols = st.columns(2)
    for i, q in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(q, key=f"ex_{i}"):
                st.session_state.example_question = q

# Check if example was clicked
if 'example_question' in st.session_state:
    user_question = st.session_state.example_question
    del st.session_state.example_question

# Generate button
if st.button("🚀 Generate SQL", type="primary", disabled=not user_question):
    with st.spinner("Generating SQL query..."):
        # For demo purposes, show a simple response
        st.success("SQL query generated!")
        q_vec = compute_embedding(embed_model, user_question)
        item = {
            "question": user_question,
            "embedding": q_vec
        }
        result = rag_query("departmetn_store", item, 3)
        sql_pred = result["generated_sql"]
        
        # Display the generated SQL
        st.code(sql_pred, language='sql')
        
        # Save to history
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
            
        st.session_state.query_history.append({
            "question": user_question,
            "sql": sql_pred,
        })
        
        # Option to execute (placeholder)
        if st.button("▶️ Execute Query"):
            st.info("Query execution would show results here")

# Query history
if 'query_history' in st.session_state and st.session_state.query_history:
    st.header("📜 Query History")
    for i, item in enumerate(reversed(st.session_state.query_history[-5:])):  # Show last 5
        with st.expander(f"Query {len(st.session_state.query_history) - i}: {item['question'][:50]}..."):
            st.write(f"**Question:** {item['question']}")
            st.code(item['sql'], language='sql')

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Spider RAG System")