import logging


def setup_logging():
    logging.basicConfig(
        format='[%(levelname)s] - %(message)s',
        datefmt='%H:%M',  # This sets the time format to just hours and minutes
        level=logging.INFO
    )
