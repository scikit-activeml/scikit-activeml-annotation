
import diskcache as dc

from paths import DATA_DISPLAY_CACHE_PATH

cache = dc.Cache(DATA_DISPLAY_CACHE_PATH)


def get_or_default(key, default) -> dict:
    if key in cache:
        return cache[key]
    cache[key] = default
    return cache[key]

