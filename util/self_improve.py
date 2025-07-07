import json
import hashlib
from typing import List
from util.process_sql import Schema, get_schema, get_sql
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