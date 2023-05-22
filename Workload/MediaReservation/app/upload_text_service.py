import datetime
from fastapi import FastAPI
from pathlib import Path
from typing import Union

# Import OpenTelemetry and Logger modules
from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())


def UploadText(req_id, text):
    """
    Dummy since we are not using memcached
    """
    global logger

    start_time = datetime.datetime.now()

    # seems to be dummy when you are not using memcached

    end_time = datetime.datetime.now()

    # logger.info(f"UploadText {req_id} {start_time.timestamp()} {end_time.timestamp()}"
    #             f" {(end_time - start_time).total_seconds()}")

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
    # logger.info(f"req_id: {req_id}, text: {text}, carrier: {carrier}")

    # 3. Call the service
    res = UploadText(req_id, text)

    return res
