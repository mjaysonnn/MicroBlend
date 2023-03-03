import base64
import configparser
import logging
import os
import pickle
import random
import string
import zlib
from datetime import datetime, timezone
from pathlib import Path

import colorlog


def get_logger(log_file_path, logger_name='myLogger'):
    """Log plain text to file and to terminal with colors"""

    logger_to_return = logging.getLogger('playground')

    # Log to file (but not to terminal)
    logfile_handler = logging.FileHandler(log_file_path)
    plain_formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)s %(message)s')
    logfile_handler.setFormatter(plain_formatter)
    logfile_handler.setLevel(logging.DEBUG)

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
    terminal_handler.setLevel(logging.DEBUG)
    terminal_handler.setFormatter(color_formatter)

    # Add handlers to logger_to_return
    logger_to_return.addHandler(logfile_handler)
    logger_to_return.addHandler(terminal_handler)
    logger_to_return.setLevel(logging.DEBUG)

    return logger_to_return


def init_logger(directory):
    """
    Init logger
    """

    # Make log directory if it doesn't exist
    # log_dir = Path(__file__).parent.absolute() / 'logs'
    log_dir = directory / 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Write log to file
    log_file_path = log_dir / (Path(__file__).stem + '.log')
    logger = get_logger(log_file_path=log_file_path, logger_name=Path(__file__).stem)

    return logger


def fetch_conf_ini():
    """
    Fetch config.ini
    """

    conf_dict = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent.absolute() / 'utils' / 'config.ini'
    # print(config_path)
    conf_dict.read(config_path)
    conf_dict = {sect: dict(conf_dict.items(sect)) for sect in conf_dict.sections()}
    conf_dict.pop('root', None)

    return conf_dict


# Encode
def native_object_encoded(x):
    """
    Encode a native object to a string
    """
    x = pickle.dumps(x)
    x = zlib.compress(x)
    x = base64.b64encode(x).decode().replace('/', '*')

    return x


# Decode
def native_object_decoded(s):
    """
    Decode a string that was encoded with native_object_encoded
    """
    s = base64.b64decode(s.replace('*', '/'))
    s = zlib.decompress(s)
    s = pickle.loads(s)
    return s


def get_timestamp_ms():
    """
    Get current timestamp in milliseconds
    """
    return int(round(datetime.now(timezone.utc).timestamp() * 1000))


def get_random_string(length):
    """
    Get random string
    """
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return result_str


def main():
    pass


if __name__ == '__main__':
    main()
