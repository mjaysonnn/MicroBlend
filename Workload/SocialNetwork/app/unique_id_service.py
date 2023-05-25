import datetime
import os
import sys
from pathlib import Path
from typing import Union

from fastapi import FastAPI

sys.path.append(os.path.join(sys.path[0], "gen-py"))
from social_network.ttypes import *

from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())


def ComposeUniqueId(req_id: int, post_type: PostType, carrier: {}) -> int:
    """
    Compose Unique ID for PostService

    Steps:
    1. Start OpenTelemetry Tracer (Start Span)
    2. Return current timestamp in milliseconds
    3. End Span
    """

    start_time = datetime.datetime.now()

    post_id = utils.get_timestamp_ms()

    end_time = datetime.datetime.now()

    return post_id


@app.get("/unique_id_service/{input_p}")
def run_unique_id(input_p: Union[str, None] = None):
    """Parse and generate input for ComposePostService
    1. Decode the input
    2. Call ComposeUniqueId function
    3. Return unique ID
    Args:
        input_p (dict): [encoded string] or [None]
    """

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    req_id = parsed_inputs.get("req_id", 3)
    post_type = parsed_inputs.get("post_type", 0)
    carrier = parsed_inputs.get("carrier", {})

    return ComposeUniqueId(req_id, post_type, carrier)
