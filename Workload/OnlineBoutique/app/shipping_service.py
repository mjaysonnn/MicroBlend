import dataclasses
import random
import string
from fastapi import FastAPI
from pathlib import Path
from typing import Union

from utils import utils

# Import OpenTelemetry and Logger modules

# Init FastAPI Application
app = FastAPI()

# OpenTelemetry Tracer

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()


@dataclasses.dataclass
class Address:
    street: str
    city: str
    state: str
    zipcode: int


def generate_random_number():
    return random.randrange(10000, 100000)


def generate_random_string():
    letters = string.ascii_lowercase
    numbers = string.digits
    return ''.join(random.choices(letters + numbers, k=20))


# return 3 digit random letter and number
def generate_TrackingID():
    return f"{generate_random_number()}-{generate_random_string()}-{generate_random_string()}-{generate_random_number()}"


# function to take an address and return a 10 digit random number
def ShippingConfirmation(address, cart):
    # parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    # with tracer.start_as_current_span("ShippingConfirmation", parent_ctx, kind=trace.SpanKind.SERVER):
    # logger.info("[ShipOrder] received request")
    TransactionID = generate_TrackingID()
    # logger.info("[ShipOrder] completed request")

    return TransactionID


@app.get("/shipping_service/{input_p}")
def run_shipping_service(input_p: Union[str, None] = None):
    # 1. Decode the input

    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}
    # logger.info(f"[ShippingService] received input: {parsed_inputs}")

    address = parsed_inputs.get("address", Address("3 New York", "NYC", "NY", 10001))
    cart = parsed_inputs.get("cart", {"OLJCESPC7Z", 1})
    # carrier = parsed_inputs.get("carrier", {})

    # 2. Run the service
    shipping_id = ShippingConfirmation(address, cart)

    return shipping_id
