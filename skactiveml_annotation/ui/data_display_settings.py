import diskcache

import skactiveml_annotation.paths as sap

cache = diskcache.Cache(sap.DATA_DISPLAY_CACHE_PATH)

def get_or_default(key, default) -> dict:
    if key in cache:
        return cache[key]
    cache[key] = default
    return cache[key]

