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


def ComposeCreatorWithUserId(
        req_id: int, user_id: int, username: str, carrier: dict
) -> Creator:
    """
    Compose Creator with User ID for PostService
    """

    start_time = datetime.datetime.now()

    creator_res = Creator(user_id=user_id, username=username)

    end_time = datetime.datetime.now()

    return creator_res


@app.get("/user_service/{input_p}")
def run_user_service(input_p: Union[str, None] = None):
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    req_id = parsed_inputs.get("req_id", 3)
    user_id = parsed_inputs.get("user_id", 2)
    username = parsed_inputs.get("username", "test")
    carrier = parsed_inputs.get("carrier", {})

    creator_res: Creator = ComposeCreatorWithUserId(req_id, user_id, username, carrier)

    return creator_res
