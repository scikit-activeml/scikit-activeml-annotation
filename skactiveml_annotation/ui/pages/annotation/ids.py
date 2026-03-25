from skactiveml_annotation.util.utils import make_ids

ids = make_ids(__name__)

KEYBOARD = ids('keyboard')

# --- Data ---
TIME_STAMP = ids('time-stamp')

# --- Containers ---
DATA_DISPLAY_CONTAINER = ids('data-display-container')
LABELS_CONTAINER = ids('labels-container')

# --- Triggers---
ANNOTATION_INIT = ids('annotation-init')
UI_TRIGGER = ids('ui-trigger')
QUERY_TRIGGER = ids('query-trigger')
START_TIME_TRIGGER = ids('start-time-trigger')
AUTO_PLAYBACK_TRIGGER = ids("auto-playback-trigger")

# --- Buttons ---
SKIP_BATCH_BTN = ids('skip-batch-btn')
ACTION_BTN = ids('action-buttons')
APPLY_PRESENTION_SETTINGS_BTN = ids('apply-presentation-settings-btn')

# --- Inputs ---
BATCH_SIZE_INPUT = ids('batch-size-input')
SUBSAMPLING_INPUT = ids('subsampling_input')
LABEL_CHIPS_INPUT = ids('label_chips_input')

# Annotation Progress
ANNOT_PROGRESS = ids('annot-progress')
NUM_SAMPLES_TEXT = ids('num_samples_text')
ANNOT_PROGRESS_TEXT = ids('annot-progress-text')

COMPUTING_OVERLAY = ids('computing-overlay')

# Data Display
DATA_PRESENTATION_SETTINGS_CONTAINER = ids('data-presentation-settings-container')
DATA_DISPLAY_CFG_DATA = ids('data-display-cfg-data')

# --- Data Presentation Settings Inputs ---
DATA_PRESENTATION_INPUT = ids('data-presentation-input')

# --- Label setting Modal ---
LABEL_SETTING_BTN = ids('label-setting-btn')
LABEL_SETTING_CONFIRM_BTN = ids('label-setting-confirm-btn')
LABEL_SETTING_SHOW_PROBAS = ids('label-setting-show-probas')
LABEL_SETTING_SORTBY = ids('label-setting-sortby')

# Auto Annotate Modal
AUTO_ANNOTATE_BTN = ids('auto-annotate-btn')
AUTO_ANNOTATE_CONFIRM_BTN = ids('auto-annotate-confirm-btn')
AUTO_ANNOTATE_THRESHOLD = ids('auto-annotate-threshold')

# Add Label
ADD_CLASS_BTN = ids('add-class-btn')
ADDED_CLASS_NAME = ids('added-class-name')

# Search Input
LABEL_SEARCH_INPUT = ids('label-search-input')
