import datetime
import os
import sys
from pathlib import Path
from typing import Union

from fastapi import FastAPI

sys.path.append(os.path.join(sys.path[0], "gen-py"))
from social_network.ttypes import *

from utils import utils, utils_social_network

# Mongo
import pymongo
from pymongo import MongoClient

test_input = utils_social_network.generate_post_class_input()

app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# MongoClient
mongo_client = None
post_collection = None


def start_mongo_client_pool():
    global mongo_client
    global post_collection

    if mongo_client is None:
        mongo_port = 27017
        mongo_client = MongoClient(
            "post-storage-mongodb", mongo_port, waitQueueTimeoutMS=10000
        )
    post_db = mongo_client["post"]
    post_collection = post_db["post"]

    # Create index
    post_collection.create_index(
        [("post_id", pymongo.ASCENDING)], name="post_id", unique=True
    )


start_mongo_client_pool()


def StorePost(req_id: int, post: Post, carrier: dict) -> None:
    global mongo_client, post_collection

    start_time = datetime.datetime.now()

    post_id = post.post_id
    author = {"user_id": post.creator.user_id, "username": post.creator.username}
    text = post.text
    medias = [
        {"media_id": post.media[i].media_id, "media_type": post.media[i].media_type}
        for i in range(len(post.media))
    ]
    post_timestamp = post.timestamp
    post_type = post.post_type
    post_to_insert = {
        "post_id": post_id,
        "author": author,
        "text": text,
        "medias": medias,
        "timestamp": post_timestamp,
        "post_type": post_type,
    }

    post_collection.update_one(
        {"post_id": post_to_insert.get("post_id")},
        {"$set": post_to_insert},
        upsert=True,
    )

    end_time = datetime.datetime.now()


@app.get("/post_storage_service/{input_p}")
def run_post_storage_service_on_vm(input_p: Union[str, None] = None):
    """
    Run PostStorageService on VM
    """

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # Parameter
    req_id = parsed_inputs.get("req_id", 3)
    post = parsed_inputs.get("post", test_input)
    carrier = parsed_inputs.get("carrier", {})

    StorePost(req_id, post, carrier)
