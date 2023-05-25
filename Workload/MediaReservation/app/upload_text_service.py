import datetime
from pathlib import Path
from typing import Union

from fastapi import FastAPI

from utils import utils

app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())


def UploadText(req_id, text):
    """
    Generate Text
    """

    start_time = datetime.datetime.now()

    # seems to be dummy when you are not using memcached, making random text
    text = utils.get_random_string(5)

    end_time = datetime.datetime.now()

    return text


@app.get("/text_service/{input_p}")
def run_text_service(input_p: Union[str, None] = None):
    """
    Upload Text Service
    """
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}
    # logger.info(f"Input: {parsed_inputs}")

    # 2. Extract the inputs
    req_id = parsed_inputs.get("req_id", 1232)
    text = parsed_inputs.get("text", "Hello World")
    return UploadText(req_id, text)
