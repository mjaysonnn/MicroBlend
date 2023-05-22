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
import requests
import sys
import urllib
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pathlib import Path
from typing import Union

# from opentelemetry import trace
# from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Fetch Thrift Format for ComposePostService
sys.path.append(os.path.join(sys.path[0], 'gen-py'))
from social_network.ttypes import *

# Import OpenTelemetry and Logger modules
from utils import utils

# Init FastAPI Application
app = FastAPI()

# OpenTelemetry Tracer
# tracer = utils_opentelemetry.set_tracer()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()


def invoke_url_shorten_service(req_id, urls, carrier):
    """
    Invoke UrlShortenService
    :param req_id:
    :param urls:
    :param carrier: for OpenTelemetry
    :return:
    """
    # # logger.info(f"req_id : {req_id} - start invoke url_shorten_service {get_timestamp_ms()}")
    # logger.info(f"TextService Invoke UrlShortenService Start {req_id} {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id, "urls": urls, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    # url_shorten_service is container name
    url = f"http://url-shorten-service:5004/url_shorten_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()  # Raise exception if status code is not 200
    res = resp.json()

    # # logger.info(f"TextService Invoke UrlShortenService End {req_id} {get_timestamp_ms()}")
    # # logger.info(f"req_id : {req_id} - end invoke url_shorten_service {get_timestamp_ms()}")
    return res


def invoke_user_mention_service(req_id, usernames, carrier):
    """
    Invoke UserMentionService
    :param req_id:
    :param usernames:
    :param carrier:
    :return:
    """
    # # logger.info(f"req_id : {req_id} - start invoke user_mention_service {get_timestamp_ms()}")
    # logger.info(f"TextService Invoke UserMentionService Start {req_id} {utils.get_timestamp_ms()}")

    input_d = {"req_id": req_id, "usernames": usernames, "carrier": carrier}
    encoded_input = utils.native_object_encoded(input_d)

    # user_mention_service is container name
    url = f"http://user-mention-service:5005/user_mention_service/{encoded_input}"
    resp = requests.get(url)
    resp.raise_for_status()
    res = resp.json()

    return res


def ComposeText(req_id: int, text: str, carrier: dict) -> TextServiceReturn:
    global logger

    # Start OpenTelemetry Tracer - If there is parent context, use it
    # parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    # with tracer.start_as_current_span("ComposeText", parent_ctx, kind=trace.SpanKind.SERVER):
    # logger.info(f"TextService Start {req_id} {utils.get_timestamp_ms()}")
    start_time = datetime.datetime.now()

    text = urllib.parse.unquote(text)  # Note: if @ does not change to %40u -> use urllib.parse.unquote(text)

    # Search and add usernames
    usernames = list()
    str_reg = '@[a-zA-Z0-9-_]+'
    match = re.findall(str_reg, text)
    for i in range(len(match)):
        usernames.append(match[i].lstrip("@"))
    # logger.debug(usernames)

    # Search and URL
    urls = list()
    url_reg = "(http://|https://)([a-zA-Z0-9-_]+)"
    match2 = re.findall(url_reg, text)
    for i in range(len(match2)):
        urls.append(match2[i][0] + match2[i][1])

    # inject writer for parent span.
    # TraceContextTextMapPropagator().inject(carrier)

    # Invoke UrlShortenService and UserMentionService
    with ThreadPoolExecutor() as executor:
        url_shorten_future = executor.submit(invoke_url_shorten_service, req_id, urls, carrier)
        user_mention_future = executor.submit(invoke_user_mention_service, req_id, usernames, carrier)

        # Make a post
        urls_res = url_shorten_future.result()
        user_mentions_res = user_mention_future.result()

    end_time = datetime.datetime.now()
    # logger.info(f"TextService {req_id} {start_time.timestamp()} {end_time.timestamp()}"
    #             f" {(end_time - start_time).total_seconds()}")

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
    # logger.debug("parsed_inputs: {}".format(parsed_inputs))

    # Parameters
    req_id = parsed_inputs.get('req_id', 3)
    text = parsed_inputs.get('text', 'testtesttesttest')
    carrier = parsed_inputs.get('carrier', {})

    # logger.info(f"Call TextService Start {req_id}  {utils.get_timestamp_ms()}")
    res = ComposeText(req_id, text, carrier)
    # # logger.info(f"Call TextService End {req_id} {get_timestamp_ms()}")
    # logger.debug(res)
    # logger.debug(type(res))
    # text_res = native_object_encoded(res)

    return res
