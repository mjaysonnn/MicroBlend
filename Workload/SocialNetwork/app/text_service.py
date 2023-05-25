"""Parse and generate input for ComposePostService
Process
 =========
1. Parse input
2. Invoke UrlShortenService and UserMentionService
3. Return result
"""
import datetime
import os
import re
import sys
import urllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import requests
from fastapi import FastAPI

sys.path.append(os.path.join(sys.path[0], "gen-py"))
from social_network.ttypes import *

from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())


def invoke_url_shorten_service(req_id, urls, carrier):
    """
    Invoke UrlShortenService
    :param req_id:
    :param urls:
    :param carrier: for OpenTelemetry
    :return:
    """

    input_d = {"req_id": req_id, "urls": urls, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    # url_shorten_service is container name
    url = f"http://url-shorten-service:5004/url_shorten_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()  # Raise exception if status code is not 200
    return resp.json()


def invoke_user_mention_service(req_id, usernames, carrier):
    """
    Invoke UserMentionService
    :param req_id:
    :param usernames:
    :param carrier:
    :return:
    """

    input_d = {"req_id": req_id, "usernames": usernames, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    # user_mention_service is container name
    url = f"http://user-mention-service:5005/user_mention_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def ComposeText(req_id: int, text: str, carrier: dict) -> TextServiceReturn:
    start_time = datetime.datetime.now()

    text = urllib.parse.unquote(
        text
    )  # Note: if @ does not change to %40u -> use urllib.parse.unquote(text)

    str_reg = "@[a-zA-Z0-9-_]+"
    match = re.findall(str_reg, text)
    usernames = [match[i].lstrip("@") for i in range(len(match))]
    url_reg = "(http://|https://)([a-zA-Z0-9-_]+)"
    match2 = re.findall(url_reg, text)
    urls = [match2[i][0] + match2[i][1] for i in range(len(match2))]

    # Invoke UrlShortenService and UserMentionService
    with ThreadPoolExecutor() as executor:
        url_shorten_future = executor.submit(
            invoke_url_shorten_service, req_id, urls, carrier
        )
        user_mention_future = executor.submit(
            invoke_user_mention_service, req_id, usernames, carrier
        )

        # Make a post
        urls_res = url_shorten_future.result()
        user_mentions_res = user_mention_future.result()

    end_time = datetime.datetime.now()

    return TextServiceReturn(text, user_mentions_res, urls_res)


@app.get("/text_service/{input_p}")
def run_text_service(input_p: Union[str, None] = None):
    """
    Parse and generate input for ComposePostService
    :param input_p:
    :return:
    """

    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # Parameters
    req_id = parsed_inputs.get("req_id", 3)
    text = parsed_inputs.get("text", "testtesttesttest")
    carrier = parsed_inputs.get("carrier", {})

    return ComposeText(req_id, text, carrier)
