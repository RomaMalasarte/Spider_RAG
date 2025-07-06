import gc
import json
import nltk
import gdown
import torch
import faiss
import spacy
import random
import pickle
import hashlib
import sqlite3
import numpy as np
import torch._dynamo
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import dataclass
from datasets import load_dataset
from collections import defaultdict
from typing import List, Dict, Tuple
from process_sql import Schema, get_schema, get_sql
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(42)
nltk.download('punkt_tab')
nlp = spacy.load("en_core_web_sm")
spider_dataset = load_dataset("spider")
torch.set_float32_matmul_precision('high')
print("Available splits:", spider_dataset.keys())

"""## GLOBALS VARIALBES

"""

# ------------- configuration -------------
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
LLM_MODEL_NAME       = "Qwen/Qwen2.5-Coder-14B-Instruct"
                       #"Qwen/Qwen2.5-Coder-14B-Instruct"
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_COLUMNS        = 10  # Number of most similar columns to include per table

# ------------- Graph configuration -------------
schema_file   = Path("spider_data/tables.json")
schema_all    = json.loads(schema_file.read_text())
schema        = next(d for d in schema_all if d["db_id"] == "department_store")
tables        = schema["table_names_original"]
columns_orig  = schema["column_names_original"]
foreign_pairs = schema["foreign_keys"]
pk_set        = set(schema["primary_keys"])
G             = nx.MultiDiGraph()

# ------------- globals (populated at runtime) -------------
TOKENIZER    = None
LLM_MODEL    = None
DOCUMENTS    : List[Dict] = []
QUERY_EMBEDS : List[Dict] = []
DOC_EMBEDS   = None
TABLE_EMBEDS = None
ROOT_TABLES  = None
COLUMN_EMBEDS = None  # New: store column embeddings separately

#-------------------------------------------------------------
Qwen_url  = 'https://drive.google.com/drive/folders/1jMWsxAKILafQaLxtb0nF28bFLoJASLSb'
Qwen = "Qwen3-Embedding-8B"

m3_url = 'https://drive.google.com/drive/folders/1KHfedpn61dmY9TEXskacFiRXueB095xc'
m3 = "BAAI-bge-m3"

large_url = 'https://drive.google.com/drive/folders/1St_sZJEQuybI6Lj0uX0wdOboS0R8-q4w'
large = "BAAI-bge-large-en-v1.5"

"""## Init"""

def initialize_rag(llm_model_name: str       = LLM_MODEL_NAME,
                   device: str | None        = None):
    """
    Load the embedding model and LLM into memory.
    Call **once** per session.
    """
    global TOKENIZER, LLM_MODEL, DEVICE
    print(f"[init] loading LLM '{llm_model_name}'  (this may take a while)…")
    TOKENIZER   = AutoTokenizer.from_pretrained(llm_model_name)
    LLM_MODEL   = AutoModelForCausalLM.from_pretrained(
                      llm_model_name,
                      torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32)
    LLM_MODEL.to(DEVICE)
    LLM_MODEL.eval()
    LLM_MODEL = torch.compile(LLM_MODEL, dynamic=True)
    print("[init] done.")

def get_random_row(cursor, table_name):
    """Original helper function to get a random row from a table."""
    try:
        # First, check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """, (table_name,))
        if not cursor.fetchone():
            print(f"Table '{table_name}' does not exist.")
            return None

        # Get total row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        if row_count == 0:
            print(f"Table '{table_name}' is empty.")
            return None

        # Get random row using RANDOM()
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1")
        random_row = cursor.fetchone()

        # Get column names for better output
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in cursor.fetchall()]

        return {
            'row': random_row,
            'columns': columns,
            'row_dict': dict(zip(columns, random_row)) if random_row else None
        }

    except Exception as e:
        print(f"Error selecting random row: {e}")
        return None

def get_all_string_values(cursor, table_name):
    """Helper function to get all possible string values from VARCHAR columns (excluding VARCHAR(255))."""
    try:
        # First, check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """, (table_name,))
        if not cursor.fetchone():
            print(f"Table '{table_name}' does not exist.")
            return None

        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()

        # Look for VARCHAR columns, but exclude VARCHAR(255)
        varchar_columns = []

        for column in columns_info:
            column_name = column[1]
            column_type = column[2].strip() if column[2] else ''
            column_type_upper = column_type.upper()

            # Check if it's a VARCHAR type
            if column_type_upper.startswith('VARCHAR'):
                # Accept plain VARCHAR
                if column_type_upper == 'VARCHAR':
                    varchar_columns.append(column_name)
                # Accept VARCHAR(n) where n != 255
                elif '(' in column_type_upper and ')' in column_type_upper:
                    # Extract the number from VARCHAR(n)
                    try:
                        start = column_type_upper.index('(') + 1
                        end = column_type_upper.index(')')
                        size = column_type_upper[start:end].strip()

                        # Include if the size is not 255
                        if size != '255':
                            varchar_columns.append(column_name)
                    except (ValueError, IndexError):
                        # If we can't parse it properly, skip it
                        pass

        if not varchar_columns:
            print(f"No suitable VARCHAR columns found in table '{table_name}'.")
            return None

        # Get all unique values for each VARCHAR column
        result = {}
        for column in varchar_columns:
            try:
                # Use quote to handle column names with special characters
                cursor.execute(f"""
                    SELECT DISTINCT "{column}"
                    FROM {table_name}
                    WHERE "{column}" IS NOT NULL
                    AND "{column}" != ''
                    ORDER BY "{column}"
                """)
                values = [row[0] for row in cursor.fetchall() if isinstance(row[0], str)]
                result[column] = values
            except Exception as e:
                print(f"Warning: Could not retrieve values for column '{column}': {e}")
                result[column] = []

        return {
            'table_name': table_name,
            'varchar_columns': varchar_columns,
            'all_string_values': result,
            'total_columns': len(varchar_columns),
            'total_unique_values': sum(len(vals) for vals in result.values())
        }

    except Exception as e:
        print(f"Error retrieving string values: {e}")
        return None


def get_random_row_with_all_strings(cursor, table_name):
    """Modified version that gets a random row AND all possible values from VARCHAR columns."""
    try:
        # First, check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """, (table_name,))
        if not cursor.fetchone():
            print(f"Table '{table_name}' does not exist.")
            return None

        # Get total row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        if row_count == 0:
            print(f"Table '{table_name}' is empty.")
            return None

        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        columns = [column[1] for column in columns_info]

        # Get random row
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1")
        random_row = cursor.fetchone()

        # Get all values from VARCHAR columns only
        varchar_values_info = get_all_string_values(cursor, table_name)

        return {
            'random_row': random_row,
            'columns': columns,
            'row_dict': dict(zip(columns, random_row)) if random_row else None,
            'all_string_values': varchar_values_info['all_string_values'] if varchar_values_info else {},
            'varchar_columns': varchar_values_info['varchar_columns'] if varchar_values_info else [],
            'total_unique_values': varchar_values_info['total_unique_values'] if varchar_values_info else 0
        }

    except Exception as e:
        print(f"Error selecting random row with string values: {e}")
        return None


def database_value(table_name,
                  file_path: str = "spider_data/database/department_store/department_store.sqlite",
                  get_all_strings: bool = True):
    """
    Access database and get random row data with optional string value collection.
    Only collects values from columns with exactly VARCHAR type (not VARCHAR(255) etc).

    Args:
        table_name: Name of the table to query
        file_path: Path to the SQLite database file
        get_all_strings: If True, also collect all possible values from VARCHAR columns

    Returns:
        Dictionary containing row data and optionally all string values
    """
    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()

        if get_all_strings:
            result = get_random_row_with_all_strings(cursor, table_name)
        else:
            result = get_random_row(cursor, table_name)

        conn.close()
        return result

    except Exception as e:
        print(f"Error accessing database: {e}")
        return None

def initialize_graph():
    global ROOT_TABLES
    print("[data] Populating the Knowledge Graph...\n")
    G.add_nodes_from(tables)
    colidx_to_tbl = {i: tables[tbl_id] for i, (tbl_id, _) in enumerate(columns_orig) if tbl_id != -1}

    # Add foreign key edges
    for child_idx, parent_idx in foreign_pairs:
        child_tbl  = colidx_to_tbl[child_idx]
        parent_tbl = colidx_to_tbl[parent_idx]
        child_col  = columns_orig[child_idx][1]
        parent_col = columns_orig[parent_idx][1]
        G.add_edge(child_tbl, parent_tbl,
                  relation="foreign_key",
                  fk=f"{parent_col} -> {child_col}")

    # Cache database values per table to avoid redundant calls
    table_data_cache = {}

    fk_child_set = {c for c, _ in foreign_pairs}         # columns that *are* FKs

    for idx, (tbl_id, col_nm) in enumerate(columns_orig):
        if tbl_id == -1:                                 # skip the dummy "*"
            continue

        tbl = tables[tbl_id]

        # Get table data only once per table
        if tbl not in table_data_cache:
            table_data_cache[tbl] = database_value(tbl)

        table_data = table_data_cache[tbl]
        col_node = f"{tbl}.{col_nm}"

        # Prepare node attributes
        node_attrs = {
            "type": "column",
            "label": col_nm,
            "pk": idx in pk_set,
            "fk": idx in fk_child_set
        }

        # Add value from random row if available
        if table_data and 'row_dict' in table_data and col_nm in table_data['row_dict']:
            node_attrs["value"] = table_data['row_dict'][col_nm]
        else:
            node_attrs["value"] = None

        # If this column is exactly VARCHAR type, add all possible values
        if (table_data and
            'varchar_columns' in table_data and
            col_nm in table_data['varchar_columns'] and
            'all_string_values' in table_data and
            col_nm in table_data['all_string_values']):
            node_attrs["all_values"] = table_data['all_string_values'][col_nm]
            node_attrs["has_all_values"] = True

        G.add_node(col_node, **node_attrs)
        G.add_edge(tbl, col_node, relation="has_column")

    # root tables = in-degree 0
    ROOT_TABLES = [t for t in tables if G.in_degree(t) == 0]
    print("[data] Knowledge Populated\n")

"""##LOAD DATA"""

def mask_nouns_spacy(text, table_names=tables, mask_token="<mask>"):
    doc = nlp(text)
    masked_tokens = []
    for token in doc:
        text = token.text.lower()
        if token.pos_ in ["NOUN", "PROPN"] and text in tables:
            masked_tokens.append(mask_token)
        else:
            masked_tokens.append(token.text)
    return " ".join(masked_tokens)

def load_data_from_file(
    max_samples: int | None = 3000,
    test_samples:int = 30,
    folder_url: str | None = None,
    folder_name: str = ".",
    ):

    global QUERY_EMBEDS, TABLE_EMBEDS, DOCUMENTS, ROOT_TABLES, COLUMN_EMBEDS
    initialize_graph()
    if folder_url:
        gdown.download_folder(folder_url, quiet=False, use_cookies=False)
    with open(f"{folder_name}/query_embeds.pkl", "rb") as f:
      QUERY_EMBEDS = pickle.load(f)
    with open(f"{folder_name}/table_embeds.pkl", "rb") as f:
      TABLE_EMBEDS = pickle.load(f)
    with open(f"{folder_name}/documents.pkl", "rb") as f:
      DOCUMENTS = pickle.load(f)
    with open(f"{folder_name}/column_embeds.pkl", "rb") as f:
      COLUMN_EMBEDS = pickle.load(f)

    if test_samples:
      QUERY_EMBEDS = QUERY_EMBEDS[:test_samples]
      #QUERY_EMBEDS = random.sample(QUERY_EMBEDS, test_samples)
      print(len(QUERY_EMBEDS))
    if max_samples:
      DOCUMENTS = DOCUMENTS[:max_samples]
      #DOCUMENTS = random.sample(DOCUMENTS, max_samples)
      print(len(DOCUMENTS))
    with open("preds_gold.sql", "w") as g:
          for item in QUERY_EMBEDS:
              g.write(f"{item['query']}\t{item['db_id']}\n")
          g.close()

def load_data(embedding_model_name: str = EMBEDDING_MODEL_NAME,
              split: str = "train",
              max_samples: int | None = None,
              test_samples: int | None = None):
    """Load the Spider dataset split, compute embeddings and build matrices."""
    print(f"[data] loading embedding model '{embedding_model_name}' on {DEVICE}…")
    EMBED_MODEL = SentenceTransformer(embedding_model_name, device=DEVICE)

    global DOCUMENTS, G, tables, TABLE_EMBEDS, QUERY_EMBEDS, ROOT_TABLES, COLUMN_EMBEDS

    DOCUMENTS.clear()
    embeds = []
    test_data = []
    data = []
    ds = load_dataset("spider", split=split)
    initialize_graph()

    # Create column embeddings with more context
    col_texts = []
    col_info = []
    for idx, (tbl_id, col_name) in enumerate(columns_orig):
        if tbl_id == -1:
            continue
        table_name = tables[tbl_id]
        # Include table context in column embedding
        col_text = f"table: {table_name}, column: {col_name}"
        col_texts.append(col_text)
        col_info.append((idx, tbl_id, col_name, table_name))

    col_vecs = EMBED_MODEL.encode(col_texts,
                                  convert_to_numpy=True,
                                  normalize_embeddings=True,
                                  batch_size=128)

    # Store column embeddings with their metadata
    COLUMN_EMBEDS = {
        'vectors': col_vecs,
        'info': col_info
    }

    # Also create table embeddings (average of column embeddings per table)
    col_tbl_ids = [tbl_id for tbl_id, _ in columns_orig]
    tbl_to_mat = defaultdict(list)
    for vec, (idx, tbl_id, col_name, table_name) in zip(col_vecs, col_info):
        tbl_to_mat[tbl_id].append(vec)
    TABLE_EMBEDS = {tid: np.stack(vs) for tid, vs in tbl_to_mat.items()}

    # Save embeddings
    with open("table_embeds.pkl", "wb") as f:
        pickle.dump(TABLE_EMBEDS, f)
    with open("column_embeds.pkl", "wb") as f:
        pickle.dump(COLUMN_EMBEDS, f)

    ROOT_TABLES = [t for t in tables if G.in_degree(t) == 0]
    print("[data] Knowledge Populated\n")
    print(f"[data] processing Spider/{split} ({len(ds)} rows)…\n")

    # Process questions
    for idx, item in enumerate(tqdm(ds, desc="Processing data")):
        vec = EMBED_MODEL.encode(
            [item['question']],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=128)[0]

        if item["db_id"] == "department_store":
            masked_question = mask_nouns_spacy(item["question"], tables)
            masked_vec = EMBED_MODEL.encode(
                [masked_question],
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=128)[0]

            QUERY_EMBEDS.append({
                "question": item["question"],
                "masked_question": masked_question,
                "query": item["query"],
                "db_id": item["db_id"],
                "embedding": vec,
                "masked_embedding": masked_vec
            })
        else:
            DOCUMENTS.append({
                "question": item["question"],
                "query": item["query"],
                "db_id": item["db_id"],
                "embedding": vec,
                "index": idx
            })

    with open("query_embeds.pkl", "wb") as f:
        pickle.dump(QUERY_EMBEDS, f)
    with open("documents.pkl", "wb") as f:
        pickle.dump(DOCUMENTS, f)

    if test_samples:
        QUERY_EMBEDS = QUERY_EMBEDS[:test_samples]
    if max_samples:
        DOCUMENTS = DOCUMENTS[:max_samples]

    with open("preds_gold.sql", "w") as g:
        for item in QUERY_EMBEDS:
            g.write(f"{item['query']}\t{item['db_id']}\n")

    print(f"[data] loaded {len(ds)} questions – ready for retrieval.")
    EMBED_MODEL.to('cpu')
    del EMBED_MODEL
    torch.cuda.empty_cache()
    gc.collect()

"""##RETRIEVE"""

def retrieve(q_vec, k: int = 5) -> List[Dict]:
    """
    Embed `question` and return the top‑`k` most similar Spider examples.
    """
    if not DOCUMENTS:
        raise RuntimeError("Call load_spider_dataset() first.")

    similarities = [q_vec @ item["embedding"].T for item in DOCUMENTS]
    idxs = np.argsort(similarities)[-k:][::-1]

    retrieved_docs = []
    for i in idxs:
        doc_with_similarity = DOCUMENTS[i].copy()
        doc_with_similarity["similarity"] = float(similarities[i])
        retrieved_docs.append(doc_with_similarity)

    return retrieved_docs

def get_top_k_columns_for_table(tbl_name: str, q_vec: np.ndarray, k: int = TOP_K_COLUMNS) -> List[Tuple[str, float, Dict]]:
    """
    Get the top-k most similar columns for a given table based on query embedding.
    Returns list of (column_name, similarity_score, column_attributes)
    """
    if COLUMN_EMBEDS is None:
        # Fallback to all columns if embeddings not available
        return [(col, 1.0, G.nodes[col]) for _, col, d in G.out_edges(tbl_name, data=True)
                if d["relation"] == "has_column"]

    # Get table index
    tbl_idx = tables.index(tbl_name)

    # Find all columns for this table
    table_columns = []
    for i, (idx, tid, col_name, table_name) in enumerate(COLUMN_EMBEDS['info']):
        if tid == tbl_idx:
            col_node = f"{tbl_name}.{col_name}"
            if col_node in G.nodes:
                similarity = float(q_vec @ COLUMN_EMBEDS['vectors'][i])
                table_columns.append((col_node, similarity, G.nodes[col_node]))

    # Sort by similarity and return top-k
    table_columns.sort(key=lambda x: x[1], reverse=True)
    return table_columns[:k]

def cols_of_table_top_k(tbl_name: str, q_vec: np.ndarray, k: int = TOP_K_COLUMNS):
    """
    Return a list of top-k most similar columns with their info.
    """
    top_columns = get_top_k_columns_for_table(tbl_name, q_vec, k)
    cols = []

    for col_node, similarity, attrs in top_columns:
        label = attrs["label"]
        value = attrs.get("value")
        tags = []

        if attrs.get("pk"):
            tags.append("PK")
        if attrs.get("fk"):
            tags.append("FK")

        info = f"{label} ({'/'.join(tags)})" if tags else label
        info = f"{info:<35}"

        # Check if we have all values for VARCHAR columns
        if attrs.get("has_all_values") and attrs.get("all_values"):
            all_vals = attrs["all_values"]
            if len(all_vals) <= 5:
                info += f"possible values: {all_vals}"
            else:
                # Show first 3 values and indicate there are more
                info += f"possible values ({len(all_vals)}): {all_vals[:3] + ['...']}"
        elif value is not None:
            # For other columns, show the sample value
            info += f"sample value: {value}"

        cols.append(info)

    return cols

def fk_edges_from(tbl_name: str):
    """
    Return each outgoing FK edge as a tuple:
        ('parent_table → child_table', 'fk_column')

    Example element:
        ('customers → orders', 'customer_id')
    """
    edges = []
    for _, parent, d in G.out_edges(tbl_name, data=True):
        if d.get("relation") == "foreign_key":
            # combined table label
            # grab the child-side column name (text before “→”)
            fk_column  = d["fk"].split('->')[0].strip()
            edge_label = f"{tbl_name}.{fk_column} -> {parent}.{fk_column}"
            edges.append((edge_label, fk_column))
    return edges

def multi_stage_search(q_vec, breadth: int = 4, max_hops: int = 4, top_n_columns: int = 3):
    """
    Breadth‑first traversal on the FK graph keeping the top‑`breadth`
    tables per layer. Uses only top N most similar columns for scoring.
    Returns a list of layers, each a list of (table_name, score) tuples.
    """
    visited: set[str] = set()
    frontier: list[str] = []
    stage_scores: list[list[tuple[float, str]]] = []

    # seed with roots
    for t in tables:
        visited.add(t)
        frontier.append(t)

    hop = 0
    while frontier and hop < max_hops:
        # score current frontier
        scored: list[tuple[float, str]] = []
        for t in frontier:
            mat = TABLE_EMBEDS[tables.index(t)]

            # Calculate similarities for all columns in the table
            similarities = mat @ q_vec

            # Get top N similarities
            if len(similarities) > top_n_columns:
                # Sort and get top N
                top_n_similarities = np.sort(similarities)[-top_n_columns:]
            else:
                # If table has fewer columns than top_n_columns, use all
                top_n_similarities = similarities

            # Use mean of top N similarities as the score
            score = float(top_n_similarities.mean())
            scored.append((score, t))

        scored.sort(reverse=True)
        stage_scores.append(scored[:breadth])

        # next frontier = FK targets of *all* tables in current frontier
        next_frontier: list[str] = []
        for _, t in scored[:breadth]:
            out_neigh = [nbr for _, nbr, d in G.out_edges(t, data=True)
                         if d["relation"] == "foreign_key"]
            for nbr in out_neigh:
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.append(nbr)

        frontier = next_frontier
        hop += 1

    return stage_scores

"""##AUGMENT / GENERATION"""

def generate_sql(
    query: Dict,
    retrieved_docs: List[Dict],
    max_length: int = 256,
    num_of_samples: int = 6,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
) -> str:
    layers = multi_stage_search(query["embedding"], breadth=4, max_hops=4)
    sys_context = (
        "You are now an excellent SQL writer. I will give you the database schema, and "
        "some SQL examples associated with similar questions that you are going to answer\n"
        "\n【Constraints】\n"
        "\t 1. You can ONLY use 'JOIN' to combine rows.\n"
        "\t 2. **NO column aliases.\n"
        "\n"
    )

    table_schema = "\n【Schema】\n"
    FK_schema = "【Foreign keys】\n"

    # Use query embedding to select top-k columns for each table
    q_vec = query["embedding"]

    for hop, layer in enumerate(reversed(layers)):
        for score, tbl in reversed(layer):
            # Get top-k columns based on similarity to query
            cols = cols_of_table_top_k(tbl, q_vec, k=TOP_K_COLUMNS)
            fks = [(txt, key) for txt, key in fk_edges_from(tbl)]

            table_schema += f"#Table: {tbl}\n"

            for idx, col in enumerate(cols):
                if idx == len(cols) - 1:
                    table_schema += f"- {col}\n\n"
                else:
                    table_schema += f"- {col},\n"

            if fks:
                for txt, key in fks:
                    FK_schema += f"{txt}\n"
    FK_schema += "-" * 60

    context = table_schema
    context += FK_schema

    context += "\n【Reference】\n"

    for i, doc in enumerate(reversed(retrieved_docs), 1):
        context += "\n"
        with open("output.txt", "a", encoding="utf-8") as f:
            print(f"similarity: {doc['similarity']}\n", file=f)
        context += f"Question: {doc['question']}\n"
        context += f"SQL: {doc['query']}\n"
    context += "-" * 60
    context += "\n###Use only the tables and columns listed in 【Schema】 and 【Foreign keys】.\n"
    context += "Think through the solution internally, then output a single-line SQLite query—no comments, no extra text, just the final SQL statement.\n"
    context += f"{query['question']}"

    messages = [
        {
          "role": "system",
          "content": f"{sys_context}",
        },
        {
            "role": "user",
            "content": f"{context}",
        }
    ]

    prompt = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt += "###SQL: "

    print("\n--- PROMPT ---")
    print(prompt)
    with open("output.txt", "a", encoding="utf-8") as f:
      print(prompt, file=f)
    print("--- END PROMPT ---\n")

    inputs = TOKENIZER(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(DEVICE)

    print(f"Input tokens: {inputs['input_ids'].shape[1]}")

    generated_samples = []

    print("\n--- Generating SQL ---")
    temp_idx = 0
    for i in range(num_of_samples):
        with torch.no_grad():
            outputs = LLM_MODEL.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                #repetition_penalty=1.1,
                pad_token_id=TOKENIZER.pad_token_id,
                eos_token_id=TOKENIZER.eos_token_id,
            )
            temp_idx += 1

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        generated_only = TOKENIZER.decode(generated_ids, skip_special_tokens=True)
        generated_only = generated_only.replace("\n", "").strip()
        print(generated_only)
        generated_samples.append(generated_only)
        print("--- END GENERATED ---\n")

    return generated_samples

def find_most_common_query_result(queries: List[str], db_path: str) -> str:
    """
    Execute every candidate SQL; return the query whose *result* appears
    most frequently (ties broken arbitrarily).
    """
    results_map, errors = {}, []

    for q in queries:
        try:
            schema  = Schema(get_schema(db_path))
            result  = get_sql(schema, q)
            h       = hashlib.md5(json.dumps(result, sort_keys=True).encode()).hexdigest()

            if h not in results_map:
                results_map[h] = {"result": result, "queries": [], "count": 0}
            results_map[h]["queries"].append(q)
            results_map[h]["count"] += 1

        except Exception as e:
            errors.append(e)

    if not results_map:
        raise RuntimeError(f"No query succeeded – errors: {errors}")

    best = max(results_map.values(), key=lambda d: d["count"])
    return best["queries"][0]

def rag_query(db_id: str, query:Dict, k: int = 3) -> Dict:
    """
    One‑liner for the full RAG pipeline:
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

"""##MAIN"""

def main(
    max_samples: int = 6912,
    test_samples: int = 2,
    folder_url: str = m3_url,
    folder_name: str = m3,
):
    global LLM_MODEL
    torch._dynamo.config.cache_size_limit = 16
    #load_data(max_samples=max_samples, test_samples=test_samples)
    load_data_from_file(max_samples=max_samples, test_samples=test_samples, folder_url=folder_url, folder_name=folder_name)
    initialize_rag()

    print("\n" + "="*60)
    print("Spider RAG System - SQL Generation")
    print("="*60 + "\n")

    with open("preds.sql", 'w') as f:
        for idx, item in enumerate(QUERY_EMBEDS):
            #torch._dynamo.reset()
            with open("output.txt", "a", encoding="utf-8") as g:
                  print(f"\n--- Processing item {idx + 1}/{len(QUERY_EMBEDS)} ---", file=g)
            print(f"\n--- Processing item {idx + 1}/{len(QUERY_EMBEDS)} ---")
            print("-" * 40)

            if isinstance(item, Dict) and 'question' in item:
                question = item['question']
            else:
                question = str(item)

            print(f"Question: {question}")

            try:
                result = rag_query("department_store", item, 1)
                sql_pred = result['generated_sql']

                print(f"\nGenerated SQL: {sql_pred}")
                f.write(f"{sql_pred}\tdepartment_store\n")

            except Exception as e:
                print(f"Error processing question: {e}\n")
                f.write(f"Error processing question: {e}\n")

    LLM_MODEL.to('cpu')
    del LLM_MODEL
    LLM_MODEL = None
    torch.cuda.empty_cache()
    gc.collect()
    print("\n" + "="*60)