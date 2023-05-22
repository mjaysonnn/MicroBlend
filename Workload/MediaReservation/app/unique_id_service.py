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
from fastapi import FastAPI
from pathlib import Path
from typing import Union

# Import OpenTelemetry and Logger modules
from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())


def UploadUniqueId(req_id: int) -> int:
    start_time = datetime.datetime.now()

    review_id = utils.get_timestamp_ms()

    end_time = datetime.datetime.now()

    # logger.info(f"UniqueIdService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
    #             f" {(end_time - start_time).total_seconds()}")

    return review_id


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

    # carrier = parsed_inputs.get('carrier', {})
    # logger.debug("req_id: {}".format(req_id))
    # logger.debug("post_type: {}".format(post_type))
    # logger.debug("carrier: {}".format(carrier))

    # 2. Generate unique ID
    unique_id = UploadUniqueId(req_id)
    # logger.debug("unique_id: {}".format(unique_id))

    # 3. Return unique ID
    return unique_id
