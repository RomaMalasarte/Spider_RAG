import torch
from typing import Dict, List
from util.retrieve import multi_stage_search, cols_of_table_top_k, fk_edges_from


def generate_user_prompt(tokenizer, query: Dict, retrieved_docs: List[Dict], TOP_K_COLUMNS: int = 5) -> str:
    """
    Generate the user prompt for SQL generation.

    Args:
        query: Dictionary containing the question and embedding
        retrieved_docs: List of similar SQL examples
        TOP_K_COLUMNS: Number of top columns to select per table

    Returns:
        The formatted user prompt string
    """
    layers = multi_stage_search(query["embedding"], breadth=4, max_hops=4)
    # System prompt
    sys_context = (
        "You are now an excellent SQL writer. I will give you the database schema, and "
        "some SQL examples associated with similar questions that you are going to answer\n"
        "\n【Constraints】\n"
        "\t 1. You can ONLY use 'JOIN' to combine rows.\n"
        "\t 2. **NO column aliases.\n"
        "\t 3. **NO YEAR() or strftime().\n"
        "\n"
    )

    # Build table schema
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

    # Build complete context
    context = table_schema
    context += FK_schema
    context += "\n【Reference】\n"

    # Add reference SQL examples
    for i, doc in enumerate(reversed(retrieved_docs), 1):
        context += "\n"
        with open("output.txt", "a", encoding="utf-8") as f:
            print(f"similarity: {doc['similarity']}\n", file=f)
        context += f"Question: {doc['question']}\n"
        context += f"SQL: {doc['query']}\n"
    context += "-" * 60

    # Add final instructions
    context += "\n###Use only the tables and columns listed in 【Schema】 and 【Foreign keys】.\n"
    context += "Think through the solution internally, then output a single-line SQLite query—no comments, no extra text, just the final SQL statement.\n\n"
    context += f"Question: {query['question']}"

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
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt += "###SQL: "
    return prompt

def generate_sql(
    tokenizer,
    llm_model,
    device: str,
    query: Dict,
    tables: List[str],
    table_embeds: List[Dict],
    column_embeds: List[Dict],
    G,
    retrieved_docs: List[Dict] | List[List[Dict]],
    max_length: int = 256,
    num_of_samples: int = 6,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
) -> List[str]:
    
    layers = multi_stage_search(
        tables=tables,
        table_embeds=table_embeds,
        G=G,
        q_vec=query["embedding"]
    )

    # Normalize retrieved_docs to always be List[List[Dict]]
    if retrieved_docs and isinstance(retrieved_docs[0], dict):
        # If it's a flat list of dicts, wrap it in another list
        retrieved_docs = [retrieved_docs]

    prompt_list = []
    input_list = []
    sample_list = []

    # Calculate sample distribution
    if num_of_samples % 2 == 1:
        half = int((num_of_samples - 1) / 2)
        sample_list = [1, half, half]
    else:
        half = int((num_of_samples - 2) / 2)
        sample_list = [1, half, half]

        # Ensure we have enough document groups for sample_list
    while len(retrieved_docs) < len(sample_list):
        retrieved_docs.append(retrieved_docs[-1])  # Duplicate last group if needed

    # Generate prompts for each document group
    for doc_group, nums in zip(retrieved_docs, sample_list):
        # doc_group is now guaranteed to be a List[Dict]
        prompt = generate_user_prompt(tokenizer, query, doc_group)
        prompt_list.append(prompt)

        # Tokenize the prompt
        input_tokens = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        # Add the tokenized input nums times to input_list
        for _ in range(int(nums)):
            input_list.append(input_tokens)

    generated_samples = []

    temp_idx = 0
    for i in range(num_of_samples):
        with torch.no_grad():
            outputs = llm_model.generate(
                **input_list[i],
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                #repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            temp_idx += 1

        input_length = input_list[i]["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        generated_only = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_only = generated_only.replace("\n", "").strip()
        generated_samples.append(generated_only)

    return generated_samples