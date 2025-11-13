"""
Wrapper for std.logging to add a new debug lvl: MYDEBUG as 
DEBUG is allready used by dependencies
"""
import logging
from typing import Final

# From logging
# CRITICAL = 50
# FATAL = CRITICAL
# ERROR = 40
# WARNING = 30
# WARN = WARNING
# INFO = 20
MYDEBUG: Final = 15
# DEBUG = 10
# NOTSET = 0

LOGGING_LVL = logging.INFO 
# For debugging
# LOGGING_LVL = MYDEBUG 

def setup_logging():
    logging.addLevelName(MYDEBUG, "DEBUG")

    logging.basicConfig(
        format='[%(levelname)s] %(message)s',
        datefmt='%H:%M',  # This sets the time format to just hours and minutes
        level=LOGGING_LVL
    )

def debug15(msg, *args, **kwargs):
    logging.log(MYDEBUG, msg, *args, **kwargs)

info = logging.info
warning = logging.warning
error = logging.error
critical = logging.critical
exception = logging.exception
debug = logging.debug
