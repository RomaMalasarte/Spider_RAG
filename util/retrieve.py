import numpy as np
import networkx as nx
from typing import List, Dict, Tuple

def retrieve(
    documents: List[Dict],
    q_vec: np.ndarray,
    k: int = 5
) -> List[Dict]:
    """
    Embed `question` and return the top‑`k` most similar Spider examples.
    """
    if k == 0:
         return []

    if not documents:
        raise RuntimeError("Call load_spider_dataset() first.")

    similarities = [q_vec @ item["embedding"].T for item in documents]
    idxs = np.argsort(similarities)[-k:][::-1]

    retrieved_docs = []
    for i in idxs:
        doc_with_similarity = documents[i].copy()
        doc_with_similarity["similarity"] = float(similarities[i])
        retrieved_docs.append(doc_with_similarity)

    return retrieved_docs

def get_top_k_columns_for_table(
    column_embeds: List[Dict],
    G: nx.MultiDiGraph,
    tables: List[str],
    tbl_name: str, 
    q_vec: np.ndarray, 
    k: int = 5
) -> List[Tuple[str, float, Dict]]:
    
    """
    Get the top-k most similar columns for a given table based on query embedding.
    Returns list of (column_name, similarity_score, column_attributes)
    """

    if column_embeds is None:
        # Fallback to all columns if embeddings not available
        return [(col, 1.0, G.nodes[col]) for _, col, d in G.out_edges(tbl_name, data=True)
                if d["relation"] == "has_column"]

    # Get table index
    tbl_idx = tables.index(tbl_name)

    # Find all columns for this table
    table_columns = []
    for i, (idx, tid, col_name, table_name) in enumerate(column_embeds['info']):
        if tid == tbl_idx:
            col_node = f"{tbl_name}.{col_name}"
            if col_node in G.nodes:
                similarity = float(q_vec @ column_embeds['vectors'][i])
                table_columns.append((col_node, similarity, G.nodes[col_node]))

    # Sort by similarity and return top-k
    table_columns.sort(key=lambda x: x[1], reverse=True)
    return table_columns[:k]

def cols_of_table_top_k(
    tbl_name: str, 
    q_vec: np.ndarray, 
    k: int = 5
) -> List[str]:
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

def multi_stage_search(
    tables: List[str],
    table_embeds: List[Dict],
    G: nx.MultiDiGraph,
    q_vec: np.ndarray, 
    breadth: int = 4, 
    max_hops: int = 4, 
    top_n_columns: int = 3
) -> List[List[Tuple[str, float]]]:
    
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
            mat = table_embeds[tables.index(t)]

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