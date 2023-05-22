import datetime
import os
import sys
from fastapi import FastAPI
from pathlib import Path
from typing import Union, List

# from opentelemetry import trace
# from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Fetch Thrift Format for ComposePostService
sys.path.append(os.path.join(sys.path[0], 'gen-py'))
from social_network.ttypes import *

# Import OpenTelemetry and Logger modules
from utils import utils

# Init FastAPI Application
app = FastAPI()

# OpenTelemetry Tracer
# tracer = utils_opentelemetry.set_tracer()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()


def ComposeMedia(req_id: int, media_types: List[str], media_ids: List[int], carrier: {}) -> List[Media]:
    """
    Compose Unique ID for PostService

    Steps:
    1. Start OpenTelemetry Tracer (Start Span)
    2. Return current timestamp in milliseconds
    3. End Span
    """
    global logger

    # parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    # with tracer.start_as_current_span("ComposeMedia", parent_ctx, kind=trace.SpanKind.SERVER):

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

    # logger.info(f"MediaService End {req_id} {utils.get_timestamp_ms()}")

    end_time = datetime.datetime.now()
    # logger.info(f"MediaService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
    # f" {(end_time - start_time).total_seconds()}")

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

    # logger.debug("input_p: {}".format(input_p))

    # 1. Decode the input
    if input_p != "Test":
        parsed_inputs = utils.native_object_decoded(input_p)
    else:
        parsed_inputs = {}
    # logger.debug("parsed_inputs: {}".format(parsed_inputs))

    req_id = parsed_inputs.get('req_id', 3)
    media_types = parsed_inputs.get('media_types', [])
    media_ids = parsed_inputs.get('media_ids', [])
    carrier = parsed_inputs.get('carrier', {})
    # logger.debug("req_id: {}".format(req_id))
    # logger.debug("post_type: {}".format(post_type))
    # logger.debug("carrier: {}".format(carrier))

    # 2. Call ComposeMedia
    media_res: List[Media] = ComposeMedia(req_id, media_types, media_ids, carrier)
    # logger.debug("unique_id: {}".format(unique_id))

    # 3. Return media_list
    return media_res

#
# @app.on_event("startup")
# async def startup_event():
#     """
#     Init logger and config.ini
#     """
