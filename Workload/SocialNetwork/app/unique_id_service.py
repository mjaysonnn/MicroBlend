"""Parse and generate input for ComposePostService
Modules
 =========

 1. `Init Logger`
 2. `Fetch config.ini`
 3. `Receive Public/Private IP`
 4. `Parse Arguments / If not (Test), make manual input`
 5. 'Generate Unique ID with OpenTelemetry'
 6. `Return Unique ID`

"""
import datetime
import os
import sys
from fastapi import FastAPI
from pathlib import Path
from typing import Union

# from opentelemetry import trace

# Fetch Thrift Format for ComposePostService
sys.path.append(os.path.join(sys.path[0], 'gen-py'))
from social_network.ttypes import *

# from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Import OpenTelemetry and Logger modules
from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()


def ComposeUniqueId(req_id: int, post_type: PostType, carrier: {}) -> int:
    """
    Compose Unique ID for PostService

    Steps:
    1. Start OpenTelemetry Tracer (Start Span)
    2. Return current timestamp in milliseconds
    3. End Span
    """
    global logger

    # Extract carrier from HTTP Request
    # parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    # with tracer.start_as_current_span("ComposeUniqueId", parent_ctx, kind=trace.SpanKind.SERVER):
    start_time = datetime.datetime.now()

    # logger.info(f"UniqueIdService Start {req_id} {utils.get_timestamp_ms()}")

    post_id = utils.get_timestamp_ms()
    # time.sleep(1)
    # logger.info(f"UniqueIdService End {req_id} {utils.get_timestamp_ms()}")

    end_time = datetime.datetime.now()
    # logger.info(f"UniqueIdService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
    #             f" {(end_time - start_time).total_seconds()}")

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
    # logger.debug("input_p: {}".format(input_p))

    req_id = parsed_inputs.get('req_id', 3)
    post_type = parsed_inputs.get('post_type', 0)
    carrier = parsed_inputs.get('carrier', {})
    # logger.debug("req_id: {}".format(req_id))
    # logger.debug("post_type: {}".format(post_type))
    # logger.debug("carrier: {}".format(carrier))

    # 2. Generate unique ID
    unique_id = ComposeUniqueId(req_id, post_type, carrier)
    # logger.debug("unique_id: {}".format(unique_id))

    # 3. Return unique ID
    return unique_id
