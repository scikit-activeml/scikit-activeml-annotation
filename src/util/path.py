from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.parent
RES_PATH = ROOT_PATH / 'res'
CONFIG_PATH = RES_PATH / 'config'
DATA_CONFIG_PATH = CONFIG_PATH / 'dataset'

SRC_PATH = ROOT_PATH / 'src'

# UI
UI_PATH = SRC_PATH / 'ui'
PAGES_PATH = UI_PATH / 'pages'