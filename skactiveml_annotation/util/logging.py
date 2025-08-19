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


def setup_logging(logging_lvl: int = MYDEBUG):
    logging.addLevelName(MYDEBUG, "MYDEBUG")

    logging.basicConfig(
        format='[%(levelname)s] - %(message)s',
        datefmt='%H:%M',  # This sets the time format to just hours and minutes
        level=logging_lvl
    )
