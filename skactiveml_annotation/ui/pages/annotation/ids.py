from dash import ALL

# TODO: Use enums to organize these ids into groups and auto assign name

ANNOTATION_INIT = 'annotation-init'
DATA_DISPLAY_CONTAINER = 'data-display-container'
LABELS_CONTAINER = 'labels-container'

# Action Buttons
ALL_ANNOTATION_BTNS = {'type': 'action-button', 'index': ALL}

UI_TRIGGER = 'ui-trigger'
QUERY_TRIGGER = 'query-trigger'
START_TIME_TRIGGER = 'start-time-trigger'
AUTO_PLAYBACK_TRIGGER = "auto-playback-trigger"

# Stats
# Annotation Progress
ANNOT_PROGRESS = 'annot-progress'
# TODo bad name NUM_SAMPLES_ANNOTATED_TEXT
NUM_SAMPLES_TEXT = 'num_samples_text'
ANNOT_PROGRESS_TEXT = 'annot-progress-text'

COMPUTING_OVERLAY = 'computing-overlay'

# Data Display
# TODO: cleanup
DATA_PRESENTATION_SETTINGS_CONTAINER = 'data-presentation-settings-container'
DATA_PRESENTATION_APPLY_BTN_CONTAINER = "data-presentation-apply-btn-container"
DATA_DISPLAY_CFG_DATA = 'data-display-cfg-data'

# --- Data Presentation Settings Inputs ---
DATA_PRESENTATION_INPUT = 'data-presentation-input'
# Image

IMAGE_RESAMPLING_METHOD_INPUT = { 'type': DATA_PRESENTATION_INPUT, 'property': 'value', 'modality': 'image', 'index': 'resampling_method' }
IMAGE_RESIZING_FACTOR_INPUT = { 'type': DATA_PRESENTATION_INPUT, 'property': 'value', 'modality': 'image', 'index': 'rescale_factor' }

# Text
TEXT_FONT_SIZE_INPUT = { 'type': DATA_PRESENTATION_INPUT, 'property': 'value', 'modality': 'text', 'index': 'font_size' }
TEXT_LINE_HEIGHT_INPUT = { 'type': DATA_PRESENTATION_INPUT, 'property': 'value', 'modality': 'text', 'index': 'line_height' }

# Audio
AUDIO_AUTOPLAY = { 'type': DATA_PRESENTATION_INPUT, 'property': 'checked', 'modality': 'audio', 'index': 'autoplay' }
AUDIO_LOOP_INPUT = { 'type': DATA_PRESENTATION_INPUT, 'property': 'checked', 'modality': 'audio', 'index': 'loop' }
AUDIO_PLAYBACK_RATE_INPUT = { 'type': DATA_PRESENTATION_INPUT, 'property': 'value', 'modality': 'audio', 'index': 'playback_rate' }

# Modals
# LABEL_SETTING_MODAL = 'label-setting-modal'
# AUTO_ANNOTATE_MODAL = 'auto-annoate-modal'
LABEL_SETTING_MODAL = { 'type': 'modal', 'index': "LabelSettingsModal"}
AUTO_ANNOTATE_MODAL = { 'type': 'modal', 'index': "AutoAnnotateModal"}

# Label setting Modal
LABEL_SETTING_BTN = 'label-setting-btn'
LABEL_SETTING_CONFIRM_BTN = 'label-setting-confirm-btn'
LABEL_SETTING_SHOW_PROBAS = 'label-setting-show-probas'
LABEL_SETTING_SORTBY = 'label-setting-sortby'

# Auto Annotate Modal
AUTO_ANNOTATE_BTN = 'auto-annotate-btn'
AUTO_ANNOTATE_CONFIRM_BTN = 'auto-annotate-confirm-btn'
AUTO_ANNOTATE_THRESHOLD = 'auto-annotate-threshold'

# Add Label
ADD_CLASS_BTN = 'add-class-btn'
ADD_CLASS_INSERTION_IDXES = 'add-class-insertion-idxes'
ADD_CLASS_WAS_ADDED = 'add-class-was-added'

# Search Input
LABEL_SEARCH_INPUT = 'label-search-input'
