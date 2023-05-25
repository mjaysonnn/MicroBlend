import datetime
from pathlib import Path
from typing import Union

from fastapi import FastAPI

from utils import utils

app = FastAPI()

logger = utils.init_logger(Path(__file__).parent.absolute())


def UploadUniqueId(req_id: int) -> int:
    """
    Compose unique ID
    """
    start_time = datetime.datetime.now()

    review_id = utils.get_timestamp_ms()

    end_time = datetime.datetime.now()

    return review_id


@app.get("/unique_id_service/{input_p}")
def run_unique_id(input_p: Union[str, None] = None):
    """
    Run unique ID service
    """
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    req_id = parsed_inputs.get("req_id", 3)

    return UploadUniqueId(req_id)
