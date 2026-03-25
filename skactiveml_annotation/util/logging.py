"""
Wrapper for std.logging to add a new debug lvl: MYDEBUG as 
DEBUG is allready used by dependencies
"""
import logging


def setup_logging():
    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        datefmt='%H:%M',  # This sets the time format to just hours and minutes
        level=logging.INFO
    )

def setup_logging_background_callback():
    # Clear existing handlers in the background process
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    setup_logging()
