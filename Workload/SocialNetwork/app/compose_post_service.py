import datetime
import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Union, List

import requests
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Fetch Thrift Format for ComposePostService
sys.path.append(os.path.join(sys.path[0], 'gen-py'))
from social_network.ttypes import *

from utils import utils, utils_opentelemetry, utils_social_network  # Import OpenTelemetry and Logger modules

app = FastAPI()  # Init FastAPI Application

tracer = utils_opentelemetry.set_tracer()  # OpenTelemetry Tracer

logger = utils.init_logger(Path(__file__).parent.absolute())  # Logging to file

conf_dict = utils.fetch_conf_ini()  # Configuration from utils/conf.ini


def invoke_text_service(req_id, text, carrier):
    """
    # 1. Get Text from TextService
    """
    # logger.info(f"ComposePostService Invoke TextService Start {req_id}  {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id, "text": text, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://text-service:5006/text_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    text_res: TextServiceReturn = resp.json()

    return text_res


def invoke_user_service(req_id, user_id, username, carrier):
    """
    2. Invoke UserService
    """
    # logger.info(f"ComposePostService Invoke UserService Start {req_id} {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id, "user_id": user_id, "username": username, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://user-service:5003/user_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()  # Raise exception if status code is not 200
    user_res = resp.json()

    return user_res


def invoke_media_service(req_id, media_types, media_ids, carrier):
    """
    3. Invoke MediaService
    """
    # logger.info(f"ComposePostService Invoke MediaService Start {req_id} {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id, "media_types": media_types, "media_ids": media_ids, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://media-service:5002/media_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    media_res = resp.json()

    return media_res


def invoke_unique_id_service(req_id, post_type, carrier):
    """
    4. Invoke UniqueIdService
    """
    # logger.info(f"ComposePostService Invoke UniqueIdService Start {req_id} {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id, "post_type": post_type, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://unique-id-service:5001/unique_id_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    unique_id_res = resp.json()

    return unique_id_res


def invoke_post_storage_service(req_id, post, carrier) -> None:
    """
    5. Invoke PostStorageService
    """
    # logger.info(f"ComposePostService Invoke PostStorageService Start {req_id} {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id, "post": post, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://post-storage-service:5008/post_storage_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()


# 6. WriteUserTimeline from UserTimelineService
def invoke_user_timeline_service(req_id, post_id, user_id, timestamp, carrier) -> None:
    """
    6. Invoke UserTimelineService
    """
    # logger.info(f"ComposePostService Invoke UserTimelineService Start {req_id} {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id, "post_id": post_id, "user_id": user_id, "timestamp": timestamp, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://user-timeline-service:5009/user_timeline_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()


def invoke_home_timeline_service(req_id, post_id, user_id, timestamp, user_mentions_id, carrier) -> None:
    """
    7. Invoke HomeTimelineService
    """
    # logger.info(f"ComposePostService Invoke HomeTimelineService Start {req_id} {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id, "post_id": post_id, "user_id": user_id, "timestamp": timestamp,
               "user_mentions_id": user_mentions_id, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://home-timeline-service:5010/home_timeline_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()


def ComposePost(req_id: int, username: str, user_id: int, text: str, media_ids: List[int],
                media_types: List[str], post_type: PostType, carrier: dict) -> None:
    """
    1. Get Text from TextService
    2. Invoke UserService
    3. Invoke MediaService
    4. Invoke UniqueIdService

    Process Post from Previous Result

    6. Invoke PostStorageService
    7. Invoke UserTimelineService
    8. Invoke HomeTimelineService
    """
    # Extract URLs from text
    parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    with tracer.start_as_current_span("ComposePost", parent_ctx, kind=trace.SpanKind.SERVER):
        # start_time = utils.get_timestamp_ms()
        start_time = datetime.datetime.now()

        TraceContextTextMapPropagator().inject(carrier)

        with ThreadPoolExecutor() as executor:  # Invoke Microservices to get variables for Post
            text_future = executor.submit(invoke_text_service, req_id, text, carrier)
            creator_future = executor.submit(invoke_user_service, req_id, user_id, username, carrier)
            media_future = executor.submit(invoke_media_service, req_id, media_types, media_ids, carrier)
            unique_id_future = executor.submit(invoke_unique_id_service, req_id, post_type, carrier)

            post_id: int = unique_id_future.result()  # Result from UniqueIdService
            # logger.debug(f"post_id is {post_id}")

            creator = creator_future.result()  # Result from UserService
            creator = Creator(**creator)

            media = media_future.result()  # Result from MediaService
            media = [Media(**i) for i in media]  # Result from MediaService
            # media = Media(**media)

            text_res = text_future.result()  # Result from TextService
            text_res = TextServiceReturn(**text_res)

            urls: List[Url] = text_res.urls  # Process Post
            # urls: List[Url] = text_res.get("urls")
            user_mentions_id = list()
            user_mentions: List[UserMention] = text_res.user_mentions
            for each_user_mention in user_mentions:
                # user_mentions_id.append(each_user_mention.user_id)
                user_mentions_id.append(each_user_mention.get("user_id"))
                # logger.debug(user_mentions_id)

        # Making Post Class
        timestamp = utils.get_timestamp_ms()
        post = Post(post_id, creator, req_id, text, user_mentions, media, urls, timestamp, post_type)
        # logger.debug(post)

        # Invoke 3 microservices
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(invoke_post_storage_service, req_id, post, carrier),
                       executor.submit(invoke_user_timeline_service, req_id, post_id, user_id, timestamp, carrier),
                       executor.submit(invoke_home_timeline_service, req_id, post_id, user_id, timestamp,
                                       user_mentions_id, carrier)]
            wait(futures)

        end_time = datetime.datetime.now()
        logger.info(f"ComposePostService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
                    f" {(end_time - start_time).total_seconds()}")


@app.get("/compose_post_service/{input_p}")
def run_compose_post_service_on_vm(input_p: Union[str, None] = None):
    """
    Parse and generate input for ComposePostService
    :param input_p:
    :return:
    """
    # For acting as Client
    # with tracer.start_as_current_span("ComposePostService Client", kind=trace.SpanKind.CLIENT):
    #     TraceContextTextMapPropagator().inject(carrier)

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}
    # logger.debug("parsed_inputs: {}".format(parsed_inputs))

    # Parameters
    compose_post_parameter = utils_social_network.generate_input_for_compose_post_service()
    req_id = parsed_inputs.get('req_id', compose_post_parameter.req_id)
    username = parsed_inputs.get('username', compose_post_parameter.username)
    user_id = parsed_inputs.get('user_id', compose_post_parameter.user_id)
    text = parsed_inputs.get('text', compose_post_parameter.text)
    media_ids = parsed_inputs.get('media_ids', compose_post_parameter.media_ids)
    media_types = parsed_inputs.get('media_types', compose_post_parameter.media_types)
    post_type = parsed_inputs.get('post_type', compose_post_parameter.post_type)
    carrier = {}

    # # logger.info(f"Call ComposePostService Start {req_id} {utils.get_timestamp_ms()}")
    ComposePost(req_id, username, user_id, text, media_ids, media_types, post_type, carrier)
    # # # logger.info(f"Call ComposePostService End {req_id} {get_timestamp_ms()}")
