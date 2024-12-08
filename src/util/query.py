from urllib.parse import parse_qs

def get_query_value(query_str: str, key: str) -> str:
    query_params = parse_qs(query_str.lstrip('?'))
    if not key in query_params:
        return None
    return query_params[key][0]


def build_query(query: dict[str, str] | tuple[str, str]) -> str:
    if isinstance(query, tuple):
        key, value = query
        return f'?{key}={value}'
    return NotImplementedError