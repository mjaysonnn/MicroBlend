"""Parse and generate input for ComposePostService
Modules
 =========

"""
import datetime
import os
import sys
from pathlib import Path
from typing import Union, List

from fastapi import FastAPI

sys.path.append(os.path.join(sys.path[0], "gen-py"))
from social_network.ttypes import *

from utils import utils

app = FastAPI()

# Mongo
import pymongo
from pymongo import MongoClient

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# MongoClient
mongo_client = None
url_shorten_collection = None


def start_mongo_client_pool():
    global mongo_client
    global url_shorten_collection

    if mongo_client is None:
        mongo_port = 27017
        mongo_client = MongoClient(
            "url-shorten-mongodb", mongo_port, waitQueueTimeoutMS=10000
        )
    post_db = mongo_client["url"]
    url_shorten_collection = post_db["url"]

    # Create index
    url_shorten_collection.create_index(
        [("expanded_url", pymongo.ASCENDING)], name="expanded_url", unique=True
    )


start_mongo_client_pool()


def ComposeUrls(req_id: int, urls: List[str], carrier: dict) -> List[Url]:
    start_time = datetime.datetime.now()

    target_urls: list[Url] = []
    for url in urls:
        url_class = Url()
        url_class.expanded_url = url
        url_class.shortened_url = f"http://short-url/{utils.get_random_string(10)}"
        target_urls.append(url_class)

    # Insert url to db
    for each_url in target_urls:
        d = {
            "expanded_url": each_url.expanded_url,
            "shortened_url": each_url.shortened_url,
        }
        url_shorten_collection.update_one(
            {"expanded_url": d.get("expanded_url")}, {"$set": d}, upsert=True
        )

    end_time = datetime.datetime.now()

    return target_urls


@app.get("/url_shorten_service/{input_p}")
def run_url_shorten_service(input_p: Union[str, None] = None):
    """Parse and generate input for ComposePostService"""

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # Parameters
    req_id = parsed_inputs.get("req_id", 3)
    urls = parsed_inputs.get(
        "urls", ["https://url_0.com", "https://url_1.com", "https://url_2.com"]
    )
    carrier = parsed_inputs.get("carrier", {})

    return ComposeUrls(req_id=req_id, urls=urls, carrier=carrier)
