import dataclasses
from pathlib import Path
from typing import Union

from fastapi import FastAPI

# Import OpenTelemetry and Logger modules
from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())


# make OrderResult dataclass
@dataclasses.dataclass
class OrderResult:
    order_id: str
    shipping_tracking_id: str
    shipping_address: str
    transaction_id: str
    cart_items: dict


def send_email_confirmation(email, order_result: OrderResult):
    """
    Send email confirmation
    """

    simple_message = {
        "from": {
            "address_spec": order_result.shipping_address,
        },
        "to": [
            {
                "address_spec": email,
            }
        ],
        "subject": "Your Confirmation Email",
        "html_body": f"Order ID: {order_result.order_id} Shipping Tracking ID: {order_result.shipping_tracking_id} Transaction ID: {order_result.transaction_id}",
    }

    print(f"Email Confirmation Sent with {simple_message}")


@app.get("/email_service/{input_p}")
def run_email_service(input_p: Union[str, None] = None):
    """
    Run Email Service
    """
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # 2. Extract the inputs
    email = parsed_inputs.get("email", "test@gmail.com")
    order_result = parsed_inputs.get(
        "order_result", OrderResult("123", "123", "123", "1234", {"123": 123})
    )

    # 3. Call the function
    send_email_confirmation(email, order_result)
