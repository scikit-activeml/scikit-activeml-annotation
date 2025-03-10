from pathlib import Path

# ROOT_PATH = Path(__file__).parent.parent.parent
ROOT_PATH = Path.cwd()

OUTPUT_PATH = ROOT_PATH / 'output'

CONFIG_PATH = ROOT_PATH / 'config'
ANNOTATED_PATH = OUTPUT_PATH / 'annotated'
CACHE_PATH = OUTPUT_PATH / 'cache'

DATA_CONFIG_PATH = CONFIG_PATH / 'dataset'
QS_CONFIG_PATH = CONFIG_PATH / 'query_strategy'

SRC_PATH = ROOT_PATH / 'src'

# UI
UI_PATH = SRC_PATH / 'ui'
PAGES_PATH = UI_PATH / 'pages'

# Datasets preconfigured dir
DATASETS_PATH = ROOT_PATH / 'datasets'
