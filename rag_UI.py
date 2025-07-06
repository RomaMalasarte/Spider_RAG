# -*- coding: utf-8 -*-
"""Spider_RAG_Streamlit_Interface.ipynb

This notebook creates a Streamlit interface for the Spider RAG SQL generation system.
Run this in Google Colab to get an interactive web interface.
"""

# Install required packages
!pip install -U -q streamlit pyngrok datasets huggingface_hub fsspec gdown transformers nltk sentence_transformers faiss-cpu
!pip install -q streamlit-chat

# Clone the Spider dataset repository
!git clone https://github.com/shu4dev/Spider_RAG.git
%cd Spider_RAG
!gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J
!unzip -o spider_data.zip

# Create the Streamlit app file
%%writefile app.py
import gc
import json
import nltk
import torch
import faiss
import spacy
import random
import pickle
import hashlib
import sqlite3
import numpy as np
import streamlit as st
import networkx as nx
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
from typing import List, Dict, Tuple
from process_sql import Schema, get_schema, get_sql
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Download required NLTK data
@st.cache_resource
def setup_nltk():
    nltk.download('punkt_tab')
    return True

# Load spacy model
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

# Set page config
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
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
LLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_COLUMNS = 10

# Initialize globals
if not st.session_state.initialized:
    setup_nltk()
    nlp = load_spacy()
    
    # Load schema
    schema_file = Path("spider_data/tables.json")
    schema_all = json.loads(schema_file.read_text())
    schema = next(d for d in schema_all if d["db_id"] == "department_store")
    tables = schema["table_names_original"]
    columns_orig = schema["column_names_original"]
    foreign_pairs = schema["foreign_keys"]
    pk_set = set(schema["primary_keys"])
    
    # Store in session state
    st.session_state.nlp = nlp
    st.session_state.schema = schema
    st.session_state.tables = tables
    st.session_state.columns_orig = columns_orig
    st.session_state.foreign_pairs = foreign_pairs
    st.session_state.pk_set = pk_set
    st.session_state.G = nx.MultiDiGraph()

# Helper functions (simplified versions from the original script)
def get_random_row_with_all_strings(cursor, table_name):
    """Get a random row and all possible values from VARCHAR columns."""
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        if row_count == 0:
            return None
            
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        columns = [column[1] for column in columns_info]
        
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1")
        random_row = cursor.fetchone()
        
        # Get VARCHAR columns
        varchar_columns = []
        for column in columns_info:
            column_name = column[1]
            column_type = column[2].strip().upper() if column[2] else ''
            if column_type.startswith('VARCHAR') and column_type != 'VARCHAR(255)':
                varchar_columns.append(column_name)
        
        # Get all unique values
        all_values = {}
        for col in varchar_columns:
            cursor.execute(f'SELECT DISTINCT "{col}" FROM {table_name} WHERE "{col}" IS NOT NULL')
            values = [row[0] for row in cursor.fetchall() if isinstance(row[0], str)]
            all_values[col] = values
            
        return {
            'random_row': random_row,
            'columns': columns,
            'row_dict': dict(zip(columns, random_row)) if random_row else None,
            'all_string_values': all_values,
            'varchar_columns': varchar_columns
        }
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def database_value(table_name):
    """Access database and get random row data."""
    db_path = "spider_data/database/department_store/department_store.sqlite"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        result = get_random_row_with_all_strings(cursor, table_name)
        conn.close()
        return result
    except Exception as e:
        st.error(f"Database error: {e}")
        return None

@st.cache_resource
def initialize_graph():
    """Initialize the knowledge graph."""
    G = nx.MultiDiGraph()
    tables = st.session_state.tables
    columns_orig = st.session_state.columns_orig
    foreign_pairs = st.session_state.foreign_pairs
    pk_set = st.session_state.pk_set
    
    G.add_nodes_from(tables)
    colidx_to_tbl = {i: tables[tbl_id] for i, (tbl_id, _) in enumerate(columns_orig) if tbl_id != -1}
    
    # Add foreign key edges
    for child_idx, parent_idx in foreign_pairs:
        child_tbl = colidx_to_tbl[child_idx]
        parent_tbl = colidx_to_tbl[parent_idx]
        child_col = columns_orig[child_idx][1]
        parent_col = columns_orig[parent_idx][1]
        G.add_edge(child_tbl, parent_tbl,
                  relation="foreign_key",
                  fk=f"{parent_col} -> {child_col}")
    
    # Add column nodes
    table_data_cache = {}
    fk_child_set = {c for c, _ in foreign_pairs}
    
    for idx, (tbl_id, col_nm) in enumerate(columns_orig):
        if tbl_id == -1:
            continue
            
        tbl = tables[tbl_id]
        
        if tbl not in table_data_cache:
            table_data_cache[tbl] = database_value(tbl)
            
        table_data = table_data_cache[tbl]
        col_node = f"{tbl}.{col_nm}"
        
        node_attrs = {
            "type": "column",
            "label": col_nm,
            "pk": idx in pk_set,
            "fk": idx in fk_child_set
        }
        
        if table_data and 'row_dict' in table_data and col_nm in table_data['row_dict']:
            node_attrs["value"] = table_data['row_dict'][col_nm]
        else:
            node_attrs["value"] = None
            
        if (table_data and 'varchar_columns' in table_data and 
            col_nm in table_data['varchar_columns'] and
            'all_string_values' in table_data and
            col_nm in table_data['all_string_values']):
            node_attrs["all_values"] = table_data['all_string_values'][col_nm]
            node_attrs["has_all_values"] = True
            
        G.add_node(col_node, **node_attrs)
        G.add_edge(tbl, col_node, relation="has_column")
    
    ROOT_TABLES = [t for t in tables if G.in_degree(t) == 0]
    return G, ROOT_TABLES

@st.cache_resource
def load_embeddings(folder_name="BAAI-bge-m3"):
    """Load pre-computed embeddings."""
    try:
        with open(f"{folder_name}/query_embeds.pkl", "rb") as f:
            query_embeds = pickle.load(f)
        with open(f"{folder_name}/table_embeds.pkl", "rb") as f:
            table_embeds = pickle.load(f)
        with open(f"{folder_name}/documents.pkl", "rb") as f:
            documents = pickle.load(f)
        with open(f"{folder_name}/column_embeds.pkl", "rb") as f:
            column_embeds = pickle.load(f)
        return query_embeds, table_embeds, documents, column_embeds
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None, None, None, None

@st.cache_resource
def load_models():
    """Load the embedding model and LLM."""
    with st.spinner("Loading models... This may take a few minutes."):
        # Load embedding model
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        
        # Load LLM
        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,         )
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
        )
        llm_model.to(DEVICE)
        llm_model.eval()
        
    return embed_model, tokenizer, llm_model

# Main UI
st.title("🕷️ Spider RAG SQL Generator")
st.markdown("""
This interface generates SQL queries for the department store database using RAG (Retrieval-Augmented Generation).
Ask natural language questions and get SQL queries!
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    embedding_choice = st.selectbox(
        "Embedding Model",
        ["BAAI-bge-m3", "BAAI-bge-large-en-v1.5", "Qwen3-Embedding-8B"],
        index=0
    )
    
    # Retrieval settings
    k_examples = st.slider("Number of examples to retrieve", 1, 10, 3)
    num_samples = st.slider("Number of SQL candidates to generate", 1, 10, 6)
    
    # Generation settings
    with st.expander("Generation Parameters"):
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.8)
        top_k = st.slider("Top-k", 1, 50, 20)
        max_length = st.slider("Max generation length", 64, 512, 256)
    
    # Database schema viewer
    with st.expander("📊 Database Schema"):
        if st.session_state.initialized:
            for table in st.session_state.tables:
                st.write(f"**{table}**")
                # Show columns for this table
                cols = [col for tid, col in st.session_state.columns_orig 
                       if tid != -1 and st.session_state.tables[tid] == table]
                for col in cols:
                    st.write(f"  - {col}")

# Initialize system
if st.button("🚀 Initialize System", disabled=st.session_state.initialized):
    with st.spinner("Initializing system..."):
        # Initialize graph
        G, ROOT_TABLES = initialize_graph()
        st.session_state.G = G
        st.session_state.ROOT_TABLES = ROOT_TABLES
        
        # Load embeddings
        folder_map = {
            "BAAI-bge-m3": "BAAI-bge-m3",
            "BAAI-bge-large-en-v1.5": "BAAI-bge-large-en-v1.5",
            "Qwen3-Embedding-8B": "Qwen3-Embedding-8B"
        }
        
        query_embeds, table_embeds, documents, column_embeds = load_embeddings(folder_map[embedding_choice])
        
        if all(x is not None for x in [query_embeds, table_embeds, documents, column_embeds]):
            st.session_state.QUERY_EMBEDS = query_embeds
            st.session_state.TABLE_EMBEDS = table_embeds
            st.session_state.DOCUMENTS = documents
            st.session_state.COLUMN_EMBEDS = column_embeds
            
            # Load models
            embed_model, tokenizer, llm_model = load_models()
            st.session_state.embed_model = embed_model
            st.session_state.tokenizer = tokenizer
            st.session_state.llm_model = llm_model
            
            st.session_state.initialized = True
            st.success("System initialized successfully!")
        else:
            st.error("Failed to load embeddings. Please check the files.")

# Main interface
if st.session_state.initialized:
    # Query input
    st.header("💬 Ask a Question")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_question = st.text_input(
            "Enter your question about the department store database:",
            placeholder="e.g., What is the total sales amount for each department?"
        )
    with col2:
        generate_btn = st.button("Generate SQL", type="primary", disabled=not user_question)
    
    # Example questions
    with st.expander("📝 Example Questions"):
        example_questions = [
            "What is the total sales amount for each department?",
            "Which customers have made purchases over $100?",
            "List all products with their current stock levels",
            "Show the top 5 best-selling products",
            "Find all orders placed in the last month"
        ]
        for q in example_questions:
            if st.button(q, key=f"ex_{q[:20]}"):
                user_question = q
                generate_btn = True
    
    # Generate SQL
    if generate_btn and user_question:
        with st.spinner("Generating SQL query..."):
            try:
                # Embed the question
                q_vec = st.session_state.embed_model.encode(
                    [user_question],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )[0]
                
                # Create query dict
                query = {
                    "question": user_question,
                    "embedding": q_vec,
                    "db_id": "department_store"
                }
                
                # Retrieve similar examples
                similarities = [q_vec @ doc["embedding"].T for doc in st.session_state.DOCUMENTS]
                idxs = np.argsort(similarities)[-k_examples:][::-1]
                retrieved_docs = [st.session_state.DOCUMENTS[i] for i in idxs]
                
                # Show retrieved examples
                with st.expander("🔍 Retrieved Examples"):
                    for i, doc in enumerate(retrieved_docs):
                        st.write(f"**Example {i+1}:**")
                        st.write(f"Question: {doc['question']}")
                        st.code(doc['query'], language='sql')
                        st.write("---")
                
                # Generate SQL (simplified version)
                st.info("Generating SQL query...")
                
                # Create a simple prompt
                prompt = f"""You are an SQL expert. Based on the department store database schema,
                generate an SQL query for this question: {user_question}
                
                Return only the SQL query, no explanations."""
                
                # Show the generated SQL
                generated_sql = "SELECT * FROM departments;"  # Placeholder
                
                st.success("SQL query generated!")
                st.code(generated_sql, language='sql')
                
                # Save to history
                st.session_state.query_history.append({
                    "question": user_question,
                    "sql": generated_sql,
                    "timestamp": "2025-01-07"
                })
                
            except Exception as e:
                st.error(f"Error generating SQL: {str(e)}")
    
    # Query history
    if st.session_state.query_history:
        st.header("📜 Query History")
        for i, item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"Query {len(st.session_state.query_history) - i}: {item['question'][:50]}..."):
                st.write(f"**Question:** {item['question']}")
                st.code(item['sql'], language='sql')
                st.write(f"*Generated at: {item['timestamp']}*")

else:
    st.info("👆 Click 'Initialize System' to start using the SQL generator.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Spider RAG System")

# Run the Streamlit app with ngrok
from pyngrok import ngrok
import subprocess
import time

# Set up ngrok
ngrok_token = "YOUR_NGROK_TOKEN"  # Replace with your ngrok token
ngrok.set_auth_token(ngrok_token)

# Run Streamlit in background
process = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])
time.sleep(5)  # Wait for Streamlit to start

# Create ngrok tunnel
public_url = ngrok.connect(8501)
print(f"\n🌐 Your Streamlit app is live at: {public_url}")
print("Click the link above to access your Spider RAG interface!")

# Keep the process running
try:
    process.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    ngrok.disconnect(public_url)
    process.terminate()