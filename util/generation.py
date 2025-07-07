import torch
from typing import Dict, List
from util.retrieve import retrieve, multi_stage_search, cols_of_table_top_k, fk_edges_from

def generate_sql(
    tokenizer,
    llm_model,
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
            cols = cols_of_table_top_k(tbl, q_vec, k=5)
            fks = [(txt, key) for txt, key in fk_edges_from(tbl)]

            table_schema += f"#Table: {tbl}\n"

            for idx, col in enumerate(cols):
                if idx == len(cols) - 1:
                    table_schema += f"- {col}\n\n"
                else:
                    table_schema += f"- {col},\n"

            if fks:
                for txt, _ in fks:
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