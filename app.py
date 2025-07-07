import torch
import streamlit as st
from typing import Dict, List
from util.retrieve import retrieve
from util.generation import generate_sql
from util.init import init_models, init_graph
from util.self_improve import find_most_common_query_result
from util.load_data import load_schema, load_data_from_file, compute_embedding

def rag_query(
    tokenizer,
    llm_model,
    documents: List[Dict],
    db_id: str,
    query: Dict, 
    k: int = 3) -> Dict:
    """
    One-liner for the full RAG pipeline:
      1. retrieve k neighbours,
      2. sample SQLs,
      3. pick the consensus execution.
    Returns a dict with the SQL and the examples retrieved.
    """

    retrieved = retrieve(
        documents=documents,
        q_vec=query["embedding"], 
        k=k
    )

    candidates = generate_sql(
        tokenizer=tokenizer, 
        llm_model=llm_model,
        device=DEVICE,
        query=query,
        G=st.session_state.G,
        tables=st.session_state.tables,
        table_embeds=st.session_state.table_embeds,
        column_embeds=st.session_state.column_embeds,
        retrieved_docs=retrieved
    )

    db_path = f"/content/Spider_RAG/spider_data/database/{db_id}/{db_id}.sqlite"
    best_sql = find_most_common_query_result(candidates, db_path)
    return {
        "question": query["question"],
        "generated_sql": best_sql,
    }

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Spider RAG SQL Generator",
    page_icon="🕷️",
    layout="wide"
)

# Global variables
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.query_history = []
    st.session_state.current_result = None

# Initialize models and data only once
if not st.session_state.initialized:
    with st.spinner("Loading models and data... This may take a few minutes on first run."):
        try:
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load schema
            schema, tables, columns_orig, foreign_pairs, pk_set = load_schema(
                schema_file="/content/Spider_RAG/spider_data/tables.json",
                db_id="department_store"
            )
            
            # Load embeddings and documents
            query_embeds, documents, table_embeds, column_embeds = load_data_from_file(
                max_samples=6912, 
                test_samples=88,
                folder_url='https://drive.google.com/drive/folders/1KHfedpn61dmY9TEXskacFiRXueB095xc',
                folder_name='BAAI-bge-m3'
            )
            
            # Initialize models with memory optimization
            tokenizer, llm_model, embed_model = init_models(
                llm_model_name=LLM_MODEL_NAME,
                embedding_model_name=EMBEDDING_MODEL_NAME,
                device=DEVICE
            )
            
            # Initialize graph
            G = init_graph(
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
            st.session_state.tokenizer = tokenizer
            st.session_state.llm_model = llm_model
            st.session_state.embed_model = embed_model
            st.session_state.documents = documents
            st.session_state.query_embeds = query_embeds
            st.session_state.table_embeds = table_embeds
            st.session_state.column_embeds = column_embeds
            
            st.session_state.initialized = True
            st.success("Models loaded successfully!")
            
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            st.stop()

# Main UI
st.title("🕷️ Spider RAG SQL Generator")
st.markdown("""
This interface generates SQL queries for the department store database using RAG (Retrieval-Augmented Generation).
Ask natural language questions and get SQL queries!
""")

# Display device info
st.sidebar.info(f"Running on: {DEVICE}")
if DEVICE == "cuda":
    st.sidebar.info(f"GPU: {torch.cuda.get_device_name(0)}")
    # Display GPU memory usage
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    st.sidebar.info(f"GPU Memory: {allocated:.2f}GB / {reserved:.2f}GB")

# Main interface
st.header("💬 Ask a Question")

# Query input
user_question = st.text_input(
    "Enter your question about the department store database:",
    placeholder="e.g., What is the total sales amount for each department?",
    key="user_question_input"
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
                st.session_state.selected_example = q
                st.rerun()

# Check if example was clicked
if 'selected_example' in st.session_state:
    user_question = st.session_state.selected_example
    del st.session_state.selected_example

# Generate button
if st.button("🚀 Generate SQL", type="primary", disabled=not user_question):
    with st.spinner("Generating SQL query..."):
        try:
            # Clear GPU cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Compute embedding for the question
            with torch.no_grad():
                q_vec = compute_embedding(st.session_state.embed_model, user_question)
            
            # Create query item
            query_item = {
                "question": user_question,
                "embedding": q_vec
            }
            
            # Generate SQL using RAG
            result = rag_query(
                tokenizer=st.session_state.tokenizer,
                llm_model=st.session_state.llm_model,
                documents=st.session_state.documents,
                db_id="department_store",  # Fixed typo: was "departmetn_store"
                query=query_item,
                k=3
            )
            
            sql_pred = result["generated_sql"]
            
            # Display success message
            st.success("SQL query generated successfully!")
            
            # Display the generated SQL
            st.subheader("Generated SQL:")
            st.code(sql_pred, language='sql')
            
            # Save to history
            if 'query_history' not in st.session_state:
                st.session_state.query_history = []
            
            st.session_state.query_history.append({
                "question": user_question,
                "sql": sql_pred,
            })
            
            # Store current result
            st.session_state.current_result = sql_pred
            
            # Clear GPU cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            st.error("GPU out of memory! Try the following:")
            st.write("1. Restart the kernel/session")
            st.write("2. Use a smaller model")
            st.write("3. Reduce batch size in generation")
            st.write("4. Switch to CPU by changing DEVICE to 'cpu'")
            
        except Exception as e:
            st.error(f"Error generating SQL: {str(e)}")
            st.write("Debug info:")
            st.write(f"- Question: {user_question}")
            st.write(f"- Device: {DEVICE}")
            if torch.cuda.is_available():
                st.write(f"- GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated")

# Execute query button (shown only if we have a result)
if st.session_state.current_result:
    if st.button("▶️ Execute Query", key="execute_query"):
        st.info("Query execution would show results here")
        # You can add actual execution logic here

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

# Add a button to clear GPU cache manually
if DEVICE == "cuda":
    if st.sidebar.button("🧹 Clear GPU Cache"):
        torch.cuda.empty_cache()
        st.sidebar.success("GPU cache cleared!")
        allocated = torch.cuda.memory_allocated() / 1024**3
        st.sidebar.info(f"Current GPU memory: {allocated:.2f}GB")