from urllib.parse import parse_qs


# TODO: This is no longer needed
def get_query_value(query_str: str, key: str) -> str | None:
    query_params = parse_qs(query_str.lstrip('?'))
    if key not in query_params:
        return None
    return query_params[key][0]


def build_query(query: dict[str, str] | tuple[str, str]) -> str:
    if isinstance(query, tuple):
        key, value = query
        return f'?{key}={value}'
    raise NotImplementedError
