from pathlib import Path

# Go up twice. Assumes src and paths.py is not moved
ROOT_PATH = Path(__file__).parent.parent

# Top Level
ASSETS_PATH = ROOT_PATH / 'assets'
CONFIG_PATH = ROOT_PATH / 'config'
DATASETS_PATH = ROOT_PATH / 'datasets'
OUTPUT_PATH = ROOT_PATH / 'output'
SRC_PATH = ROOT_PATH / 'src'

# config
EMBEDDING_CONFIG_PATH = CONFIG_PATH / 'embedding'
DATA_CONFIG_PATH = CONFIG_PATH / 'dataset'
MODEL_CONFIG_PATH = CONFIG_PATH / 'model'
QS_CONFIG_PATH = CONFIG_PATH / 'query_strategy'

# output
ANNOTATED_PATH = OUTPUT_PATH / 'annotated'
CACHE_PATH = OUTPUT_PATH / 'cache'
PROFILER_PATH = OUTPUT_PATH / 'profiler'

# output/cache
BACKGROUND_CALLBACK_CACHE_PATH = CACHE_PATH / 'background_callback_cache'
EMBEDDINGS_CACHE_PATH = CACHE_PATH / 'embeddings_cache'
DATA_DISPLAY_CACHE_PATH = CACHE_PATH / 'data_display_cache'

# src
UI_PATH = SRC_PATH / 'ui'

# src/ui
PAGES_PATH = UI_PATH / 'pages'