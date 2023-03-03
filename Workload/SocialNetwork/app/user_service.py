import datetime
import os
import sys
from pathlib import Path
from typing import Union

from fastapi import FastAPI
from opentelemetry import trace

# Fetch Thrift Format for ComposePostService
sys.path.append(os.path.join(sys.path[0], 'gen-py'))
from social_network.ttypes import *

# Import OpenTelemetry and Logger modules
from utils import utils, utils_opentelemetry

from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Init FastAPI Application
app = FastAPI()

# OpenTelemetry Tracer
tracer = utils_opentelemetry.set_tracer()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()


def ComposeCreatorWithUserId(req_id: int, user_id: int, username: str, carrier: dict) -> Creator:
    """
    Compose Creator with User ID for PostService
    """
    global logger

    parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    with tracer.start_as_current_span("ComposeCreatorWithUserId", parent_ctx, kind=trace.SpanKind.SERVER):
        start_time = datetime.datetime.now()

        creator_res = Creator(user_id=user_id, username=username)

        end_time = datetime.datetime.now()
        logger.info(f"UserService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
                    f" {(end_time - start_time).total_seconds()}")

        return creator_res


@app.get("/user_service/{input_p}")
def run_user_service(input_p: Union[str, None] = None):
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}
    # logger.debug("parsed_inputs: {}".format(parsed_inputs))

    req_id = parsed_inputs.get('req_id', 3)
    user_id = parsed_inputs.get('user_id', 2)
    username = parsed_inputs.get('username', "test")
    carrier = parsed_inputs.get('carrier', {})

    creator_res: Creator = ComposeCreatorWithUserId(req_id, user_id, username, carrier)

    # Return encoded result
    return creator_res
