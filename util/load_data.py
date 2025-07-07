import json
import torch
import gdown
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

def load_schema(
    schema_file: str = "spider_data/tables.json", 
    db_id: str = "department_store"
    ) -> Tuple[Dict, List, List, List, set]:
    """
    Load the Spider schema JSON and extract relevant structures.
    Returns (schema, tables, column_names, foreign_pairs, primary_keys).
    """
    schema_all = json.loads(Path(schema_file).read_text())
    schema = next(d for d in schema_all if d["db_id"] == db_id)
    tables = schema["table_names_original"]
    columns_orig = schema["column_names_original"]
    foreign_pairs = schema["foreign_keys"]
    pk_set = set(schema["primary_keys"])
    return schema, tables, columns_orig, foreign_pairs, pk_set

def load_data_from_file(
    max_samples: int | None = 3000,
    test_samples:int = 30,
    folder_url: str | None = None,
    folder_name: str = ".",
    )-> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:

    query_embeds : List[Dict] = []
    documents: List[Dict] = []
    table_embeds: List[Dict] = []
    column_embeds: List[Dict] = []
    
    if folder_url:
        gdown.download_folder(folder_url, quiet=False, use_cookies=False)
    with open(f"{folder_name}/query_embeds.pkl", "rb") as f:
      query_embeds = pickle.load(f)
    with open(f"{folder_name}/table_embeds.pkl", "rb") as f:
      table_embeds = pickle.load(f)
    with open(f"{folder_name}/documents.pkl", "rb") as f:
      documents = pickle.load(f)
    with open(f"{folder_name}/column_embeds.pkl", "rb") as f:
      column_embeds = pickle.load(f)

    if test_samples:
      column_embeds = column_embeds[:test_samples]
      #QUERY_EMBEDS = random.sample(QUERY_EMBEDS, test_samples)
    if max_samples:
        documents = documents[:max_samples]
        #DOCUMENTS = random.sample(DOCUMENTS, max_samples)
    
    return query_embeds, documents, table_embeds, column_embeds

def compute_embedding(
    model: SentenceTransformer,
    texts: str,
) -> np.ndarray:
    """
    Compute embeddings for a list of texts using the provided model and tokenizer.
    Returns a list of embeddings.
    """
    vector = model.encode(
       texts, 
       convert_to_numpy=True, 
       normalize_embeddings=True, 
       batch_size=128
      )
    
    return vector

