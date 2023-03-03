import datetime
from pathlib import Path
from typing import Union

from fastapi import FastAPI
from pymongo import MongoClient

# Import OpenTelemetry and Logger modules
from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()

# MongoClient
mongo_client = None
user_collection = None


def start_mongo_client_pool():
    global logger
    global mongo_client
    global user_collection

    # Init MongoClient
    mongo_port = 27017
    if mongo_client is None:
        mongo_client = MongoClient("user-storage-mongodb", mongo_port, waitQueueTimeoutMS=10000)
    post_db = mongo_client['user']
    user_collection = post_db['user']

    # Create index
    # user_collection.create_index([('username', pymongo.ASCENDING)], name='post_id', unique=True)


start_mongo_client_pool()


def RegisterUser(req_id, first_name, last_name, username, password):
    """
    Compose Creator with User ID for PostService
    """
    global logger

    start_time = datetime.datetime.now()

    user_id = int(datetime.datetime.now().timestamp())

    post_to_insert = {
        'user_id': user_id,
        'first_name': first_name,
        'last_name': last_name,
        'username': username,
        'password': password,
    }

    user_collection.update_one({'username': post_to_insert.get('username')}, {'$set': post_to_insert}, upsert=True)

    end_time = datetime.datetime.now()
    # logger.info(f"UserService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
    #             f" {(end_time - start_time).total_seconds()}")

    return user_id


@app.get("/user_service/{input_p}")
def run_user_service(input_p: Union[str, None] = None):
    # logger.debug("input_p: {}".format(input_p))

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}
    # logger.debug("parsed_inputs: {}".format(parsed_inputs))

    # 2. Extract the inputs
    req_id = parsed_inputs.get('req_id', 3)
    first_name = parsed_inputs.get('first_name', "test")
    last_name = parsed_inputs.get('last_name', "test")
    username = parsed_inputs.get('username', "test")
    password = parsed_inputs.get('password', "test")

    user_id = RegisterUser(req_id, first_name, last_name, username, password)

    return user_id
