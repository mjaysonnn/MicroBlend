import logging
import os
from pathlib import Path

import colorlog


def get_logger(log_file_path, logger_name='myLogger'):
    """Log plain text to file and to terminal with colors"""

    # Create logger object
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Log to file
    logfile_handler = logging.FileHandler(log_file_path)
    plain_formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)s %(message)s')
    logfile_handler.setFormatter(plain_formatter)
    logger.addHandler(logfile_handler)

    # Logging info level to stdout with colors
    terminal_handler = colorlog.StreamHandler()
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(asctime)s %(lineno)-6s %(blue)s%(message)s",
        datefmt='%Y-%m-%d %H:%M:%S', log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }, secondary_log_colors={})
    terminal_handler.setFormatter(color_formatter)
    logger.addHandler(terminal_handler)

    return logger


def init_logger(filename=None):
    """Logger Handler and Files"""

    # Make parent of log directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.absolute() / 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Write log to file
    result_log_with_dir = log_dir / (Path(__file__).stem + '.log')

    # Empty log file
    open(result_log_with_dir, "w").close()
    assert os.path.getsize(result_log_with_dir) == 0

    logger = get_logger(log_file_path=result_log_with_dir, logger_name=Path(__file__).stem)
    logger.info("Initialized Logger and Emptied Result Log\n")

    return logger, result_log_with_dir


def empty_log_file():
    """
    Empty log file before experiment
    """

    # Get log file path
    log_file_path = f"logs/{os.path.splitext(os.path.basename(__file__))[0]}.log"
    # Remove contents from log file
    try:
        with open(log_file_path, "w") as log_file:
            log_file.write("")
    except FileNotFoundError:
        print(f"{log_file_path} not found")
