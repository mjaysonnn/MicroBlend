import random
import string
from datetime import datetime, timezone


def get_timestamp_ms():
    return int(round(datetime.now(timezone.utc).timestamp() * 1000))


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))
    # print(result_str)
    return result_str
