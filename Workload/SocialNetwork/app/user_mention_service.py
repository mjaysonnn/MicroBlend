"""Parse and generate input for ComposePostService"""
import datetime
import os
import sys
from pathlib import Path
from typing import Union, List

from fastapi import FastAPI

sys.path.append(os.path.join(sys.path[0], "gen-py"))
from social_network.ttypes import *

from utils import utils

# Mongo
import pymongo
from pymongo import MongoClient

user_name_list = ["username_1", "username_2"]

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# MongoClient
mongo_client = None
user_collection = None


def start_mongo_client_pool():
    global mongo_client
    global user_collection

    if mongo_client is None:
        # MongoClient
        mongodb_ip_addr = "social-graph-mongodb"
        mongodb_port = 27017
        mongo_client = MongoClient(
            mongodb_ip_addr, mongodb_port, waitQueueTimeoutMS=10000
        )

    user_db = mongo_client["user"]
    user_collection = user_db["user"]

    # Index
    user_collection.create_index(
        [("username", pymongo.ASCENDING)], name="username", unique=True
    )


start_mongo_client_pool()


def ComposeUserMentions(
        req_id: int, usernames: List[str], carrier: dict
) -> List[UserMention]:
    global user_collection

    start_time = datetime.datetime.now()

    user_mention_list = []
    for each_username in usernames:
        # find user_id from mongodb collection
        post = user_collection.find_one(filter={"username": each_username})
        user_id = post.get("user_id")
        user_mention_list.append(UserMention(user_id, each_username))

    end_time = datetime.datetime.now()

    return user_mention_list


@app.get("/user_mention_service/{input_p}")
def run_user_mention_service_on_vm(input_p: Union[str, None] = None):
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # Parameter
    req_id = parsed_inputs.get("req_id", 3)
    usernames = parsed_inputs.get("usernames", user_name_list)
    carrier = parsed_inputs.get("carrier", {})
    return ComposeUserMentions(req_id=req_id, usernames=usernames, carrier=carrier)
