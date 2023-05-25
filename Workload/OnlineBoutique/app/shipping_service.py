import dataclasses
import random
import string
from pathlib import Path
from typing import Union

from fastapi import FastAPI

from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())


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
    return "".join(random.choices(letters + numbers, k=20))


# return 3 digit random letter and number
def generate_TrackingID():
    """
    Generate tracking ID
    """
    return f"{generate_random_number()}-{generate_random_string()}-{generate_random_string()}-{generate_random_number()}"


def ShippingConfirmation(address, cart):
    """
    Shipping confirmation
    """
    return generate_TrackingID()


@app.get("/shipping_service/{input_p}")
def run_shipping_service(input_p: Union[str, None] = None):
    """
    Run shipping service

    """
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    address = parsed_inputs.get("address", Address("3 New York", "NYC", "NY", 10001))
    cart = parsed_inputs.get("cart", {"OLJCESPC7Z", 1})
    return ShippingConfirmation(address, cart)
