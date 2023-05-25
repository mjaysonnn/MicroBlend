import datetime
from pathlib import Path
from typing import Union

from fastapi import FastAPI
from pymongo import MongoClient

from utils import utils

app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()

# MongoClient
mongo_client = None
user_collection = None


def start_mongo_client_pool():
    """
    Start mongo client pool
    """
    global logger
    global mongo_client
    global user_collection

    if mongo_client is None:
        mongo_port = 27017
        mongo_client = MongoClient(
            "user-storage-mongodb", mongo_port, waitQueueTimeoutMS=10000
        )
    post_db = mongo_client["user"]
    user_collection = post_db["user"]


start_mongo_client_pool()


def RegisterUser(req_id, first_name, last_name, username, password):
    """
    Compose Creator with User ID for PostService
    """

    start_time = datetime.datetime.now()

    user_id = int(datetime.datetime.now().timestamp())

    post_to_insert = {
        "user_id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "username": username,
        "password": password,
    }

    user_collection.update_one(
        {"username": post_to_insert.get("username")},
        {"$set": post_to_insert},
        upsert=True,
    )

    end_time = datetime.datetime.now()

    return user_id


@app.get("/user_service/{input_p}")
def run_user_service(input_p: Union[str, None] = None):
    """
    Run user service
    """

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # 2. Extract the inputs
    req_id = parsed_inputs.get("req_id", 3)
    first_name = parsed_inputs.get("first_name", "test")
    last_name = parsed_inputs.get("last_name", "test")
    username = parsed_inputs.get("username", "test")
    password = parsed_inputs.get("password", "test")

    return RegisterUser(req_id, first_name, last_name, username, password)
