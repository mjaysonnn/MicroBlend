import datetime
import os
import sys
from fastapi import FastAPI
from pathlib import Path
from typing import Union

# from opentelemetry import trace

# Fetch Thrift Format for ComposePostService
sys.path.append(os.path.join(sys.path[0], 'gen-py'))
from social_network.ttypes import *

# Import OpenTelemetry and Logger modules
from utils import utils, utils_social_network

# Mongo
import pymongo
from pymongo import MongoClient

# from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

test_input = utils_social_network.generate_post_class_input()

# Init FastAPI Application
app = FastAPI()

# OpenTelemetry Tracer
# tracer = utils_opentelemetry.set_tracer()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()

# MongoClient
mongo_client = None
post_collection = None


def start_mongo_client_pool():
    global logger
    global mongo_client
    global post_collection

    # Init MongoClient
    mongo_port = 27017
    if mongo_client is None:
        mongo_client = MongoClient("post-storage-mongodb", mongo_port, waitQueueTimeoutMS=10000)
    post_db = mongo_client['post']
    post_collection = post_db['post']

    # Create index
    post_collection.create_index([('post_id', pymongo.ASCENDING)], name='post_id', unique=True)


start_mongo_client_pool()


def StorePost(req_id: int, post: Post, carrier: dict) -> None:
    global logger
    global mongo_client, post_collection

    # parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    # logger.debug(type(post.creator))
    # logger.debug(type(post.media[0]))

    # with tracer.start_as_current_span("StorePost", parent_ctx, kind=trace.SpanKind.SERVER):
    start_time = datetime.datetime.now()
    # logger.info(f"PostStorageService Start {req_id} {utils.get_timestamp_ms()}")

    # Insert post to mongodb

    post_id = post.post_id
    author = {
        'user_id': post.creator.user_id,
        # 'user_id': post.creator.get('user_id'),
        'username': post.creator.username
        # 'username': post.creator.get('username')
    }
    text = post.text
    medias = list()
    for i in range(len(post.media)):
        medias.append({
            'media_id': post.media[i].media_id,
            # 'media_id': post.media[i].get('media_id'),
            'media_type': post.media[i].media_type
            # 'media_type': post.media[i].get('media_type')
        })
    post_timestamp = post.timestamp
    post_type = post.post_type
    post_to_insert = {
        'post_id': post_id,
        'author': author,
        'text': text,
        'medias': medias,
        'timestamp': post_timestamp,
        'post_type': post_type
    }

    post_collection.update_one({'post_id': post_to_insert.get('post_id')}, {'$set': post_to_insert}, upsert=True)

    # For Insanity check
    # cursor = post_collection.find({})
    # for document in cursor:
    #     logger.debug(document)

    # # logger.info(f"PostStorageService End {req_id} {utils.get_timestamp_ms()}")

    end_time = datetime.datetime.now()
    # logger.info(f"PostStorageService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
    #             f" {(end_time - start_time).total_seconds()}")


@app.get("/post_storage_service/{input_p}")
def run_post_storage_service_on_vm(input_p: Union[str, None] = None):
    """
    Run PostStorageService on VM
    """

    # logger.debug("input_p: {}".format(input_p))

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}
    # logger.debug("parsed_inputs: {}".format(parsed_inputs))

    # Parameter
    req_id = parsed_inputs.get('req_id', 3)
    post = parsed_inputs.get('post', test_input)
    # logger.debug("post: {}".format(post))
    # logger.debug(type(post))
    carrier = parsed_inputs.get('carrier', {})
    # logger.debug("req_id: {}".format(req_id))
    # logger.debug("post: {}".format(post))
    # logger.debug("carrier: {}".format(carrier))

    # Call UserMentionService
    # logger.info(f"Call PostStorage Start {req_id} {utils.get_timestamp_ms()}")
    StorePost(req_id, post, carrier)
    # # logger.info(f"Call UserMentionService End {req_id} {utils.get_timestamp_ms()}")
