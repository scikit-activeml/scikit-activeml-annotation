from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent
RES_PATH = ROOT_PATH / 'res'

CONFIG_PATH = RES_PATH / 'config'
ANNOTATED_PATH = RES_PATH / 'annotated'

DATA_CONFIG_PATH = CONFIG_PATH / 'dataset'
QS_CONFIG_PATH = CONFIG_PATH / 'query_strategy'

SRC_PATH = ROOT_PATH / 'src'

# UI
UI_PATH = SRC_PATH / 'ui'
PAGES_PATH = UI_PATH / 'pages'