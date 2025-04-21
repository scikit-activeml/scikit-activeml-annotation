
import diskcache as dc

from paths import OUTPUT_PATH

cache = dc.Cache(OUTPUT_PATH / 'frontend')


def get_or_default(key, default) -> dict:
    if key in cache:
        return cache[key]
    cache[key] = default
    return cache[key]

