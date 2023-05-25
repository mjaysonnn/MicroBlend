import datetime
import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Union, List

import requests
from fastapi import FastAPI

sys.path.append(os.path.join(sys.path[0], "gen-py"))
from social_network.ttypes import *

from utils import utils, utils_social_network

app = FastAPI()  # Init FastAPI Application

logger = utils.init_logger(Path(__file__).parent.absolute())  # Logging to file


def invoke_text_service(req_id, text, carrier):
    """
    # 1. Get Text from TextService
    """

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

    input_d = {
        "req_id": req_id,
        "user_id": user_id,
        "username": username,
        "carrier": carrier,
    }
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://user-service:5003/user_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()  # Raise exception if status code is not 200
    return resp.json()


def invoke_media_service(req_id, media_types, media_ids, carrier):
    """
    3. Invoke MediaService
    """

    input_d = {
        "req_id": req_id,
        "media_types": media_types,
        "media_ids": media_ids,
        "carrier": carrier,
    }
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://media-service:5002/media_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def invoke_unique_id_service(req_id, post_type, carrier):
    """
    4. Invoke UniqueIdService
    """

    input_d = {"req_id": req_id, "post_type": post_type, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://unique-id-service:5001/unique_id_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def invoke_post_storage_service(req_id, post, carrier) -> None:
    """
    5. Invoke PostStorageService
    """

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

    input_d = {
        "req_id": req_id,
        "post_id": post_id,
        "user_id": user_id,
        "timestamp": timestamp,
        "carrier": carrier,
    }
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://user-timeline-service:5009/user_timeline_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()


def invoke_home_timeline_service(
        req_id, post_id, user_id, timestamp, user_mentions_id, carrier
) -> None:
    """
    7. Invoke HomeTimelineService
    """

    input_d = {
        "req_id": req_id,
        "post_id": post_id,
        "user_id": user_id,
        "timestamp": timestamp,
        "user_mentions_id": user_mentions_id,
        "carrier": carrier,
    }
    encoded_input = utils.native_object_encoded(input_d)

    url = f"http://home-timeline-service:5010/home_timeline_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()


def ComposePost(
        req_id: int,
        username: str,
        user_id: int,
        text: str,
        media_ids: List[int],
        media_types: List[str],
        post_type: PostType,
        carrier: dict,
) -> None:
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

    start_time = datetime.datetime.now()

    with ThreadPoolExecutor() as executor:  # Invoke Microservices to get variables for Post
        text_future = executor.submit(invoke_text_service, req_id, text, carrier)
        creator_future = executor.submit(
            invoke_user_service, req_id, user_id, username, carrier
        )
        media_future = executor.submit(
            invoke_media_service, req_id, media_types, media_ids, carrier
        )
        unique_id_future = executor.submit(
            invoke_unique_id_service, req_id, post_type, carrier
        )

        post_id: int = unique_id_future.result()  # Result from UniqueIdService

        creator = creator_future.result()  # Result from UserService
        creator = Creator(**creator)

        media = media_future.result()  # Result from MediaService
        media = [Media(**i) for i in media]  # Result from MediaService

        text_res = text_future.result()  # Result from TextService
        text_res = TextServiceReturn(**text_res)

        urls: List[Url] = text_res.urls  # Process Post
        user_mentions: List[UserMention] = text_res.user_mentions
        user_mentions_id = [
            each_user_mention.get("user_id") for each_user_mention in user_mentions
        ]
    # Making Post Class
    timestamp = utils.get_timestamp_ms()
    post = Post(
        post_id, creator, req_id, text, user_mentions, media, urls, timestamp, post_type
    )

    # Invoke 3 microservices
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(invoke_post_storage_service, req_id, post, carrier),
            executor.submit(
                invoke_user_timeline_service,
                req_id,
                post_id,
                user_id,
                timestamp,
                carrier,
            ),
            executor.submit(
                invoke_home_timeline_service,
                req_id,
                post_id,
                user_id,
                timestamp,
                user_mentions_id,
                carrier,
            ),
        ]
        wait(futures)

    end_time = datetime.datetime.now()


@app.get("/compose_post_service/{input_p}")
def run_compose_post_service_on_vm(input_p: Union[str, None] = None):
    """
    Parse and generate input for ComposePostService
    """

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # Parameters
    compose_post_parameter = (
        utils_social_network.generate_input_for_compose_post_service()
    )
    req_id = parsed_inputs.get("req_id", compose_post_parameter.req_id)
    username = parsed_inputs.get("username", compose_post_parameter.username)
    user_id = parsed_inputs.get("user_id", compose_post_parameter.user_id)
    text = parsed_inputs.get("text", compose_post_parameter.text)
    media_ids = parsed_inputs.get("media_ids", compose_post_parameter.media_ids)
    media_types = parsed_inputs.get("media_types", compose_post_parameter.media_types)
    post_type = parsed_inputs.get("post_type", compose_post_parameter.post_type)
    carrier = {}

    ComposePost(
        req_id, username, user_id, text, media_ids, media_types, post_type, carrier
    )
