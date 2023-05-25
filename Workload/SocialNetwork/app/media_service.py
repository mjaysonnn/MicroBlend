import datetime
import os
import sys
from pathlib import Path
from typing import Union, List

from fastapi import FastAPI

# Fetch Thrift Format for ComposePostService
sys.path.append(os.path.join(sys.path[0], "gen-py"))
from social_network.ttypes import *

from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())


def ComposeMedia(
        req_id: int, media_types: List[str], media_ids: List[int], carrier: {}
) -> List[Media]:
    """
    Compose Unique ID for PostService

    Steps:
    1. Start OpenTelemetry Tracer (Start Span)
    2. Return current timestamp in milliseconds
    3. End Span
    """

    start_time = datetime.datetime.now()

    try:
        len(media_types) == len(media_ids)
    except ServiceException:
        logger.error("The lengths of media_id list and media_type list are not equal")

    # Append Media id and Media Types
    media_list = []
    for i in range(len(media_ids)):
        new_media = Media(media_id=media_ids[i], media_type=media_types[i])
        media_list.append(new_media)

    end_time = datetime.datetime.now()

    return media_list


@app.get("/media_service/{input_p}")
def run_media_service(input_p: Union[str, None] = None):
    """Parse and generate input for ComposePostService
    1. Decode the input
    2. Call ComposeMedia
    3. Return media_list
    Args:
        input_p (dict): [encoded string] or [None]
    """

    # 1. Decode the input
    if input_p != "Test":
        parsed_inputs = utils.native_object_decoded(input_p)
    else:
        parsed_inputs = {}

    req_id = parsed_inputs.get("req_id", 3)
    media_types = parsed_inputs.get("media_types", [])
    media_ids = parsed_inputs.get("media_ids", [])
    carrier = parsed_inputs.get("carrier", {})

    # 2. Call ComposeMedia
    media_res: List[Media] = ComposeMedia(req_id, media_types, media_ids, carrier)

    # 3. Return media_list
    return media_res
