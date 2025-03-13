import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum logging level (DEBUG, INFO, etc.)
    format='[%(levelname)s] in %(filename)s: %(message)s',  # Format for log messages
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)

logger = logging.getLogger(__name__)
