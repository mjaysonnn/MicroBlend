import random
import string
from datetime import datetime, timezone


def get_timestamp_ms():
    """
    Returns the current timestamp in milliseconds.
    """
    return int(round(datetime.now(timezone.utc).timestamp() * 1000))


def get_random_string(length):
    """
    Returns a random string of lowercase letters of length `length`.
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))
