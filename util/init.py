
import json
import torch
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple
from util.load_data import database_value
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

def init_models(
    llm_model_name: str,
    embedding_model_name: str,
    device: str
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, SentenceTransformer]:
    """
    Initialize the LLM and the embedding model. Returns (tokenizer, llm_model, embed_model).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_name,
        trust_remote_code=True
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    llm_model.to(device)
    llm_model.eval()
    # Torch compile for performance
    llm_model = torch.compile(llm_model, dynamic=True)

    embed_model = SentenceTransformer(embedding_model_name)
    return tokenizer, llm_model, embed_model

def init_graph(
    tables: List[str],
    columns_orig: List[Tuple[int, str]],
    foreign_pairs: List[Tuple[int, int]],
    pk_set: set,
    G: nx.MultiDiGraph = nx.MultiDiGraph()
)-> nx.MultiDiGraph:
    
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
    return G