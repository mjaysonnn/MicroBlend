import dataclasses
from fastapi import FastAPI
from pathlib import Path
from typing import Union

# Import OpenTelemetry and Logger modules
from utils import utils

# from opentelemetry import trace
# from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Init FastAPI Application
app = FastAPI()

# OpenTelemetry Tracer
# tracer = utils_opentelemetry.set_tracer()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Configuration from utils/conf.ini
conf_dict = utils.fetch_conf_ini()


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
    UploadMovieReview
    """
    global logger

    # parent_ctx = TraceContextTextMapPropagator().extract(carrier) if carrier else {}

    # with tracer.start_as_current_span("EmailConfirmation", parent_ctx, kind=trace.SpanKind.SERVER):
    simple_message = {
        "from": {
            "address_spec": order_result.shipping_address,
        },
        "to": [{
            "address_spec": email,
        }],
        "subject": "Your Confirmation Email",
        "html_body": "Order ID: " + order_result.order_id +
                     " Shipping Tracking ID: " + order_result.shipping_tracking_id +
                     " Transaction ID: " + order_result.transaction_id
    }

    # logger.info(f"Email Confirmation Sent with {simple_message}")


@app.get("/email_service/{input_p}")
def run_email_service(input_p: Union[str, None] = None):
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # 2. Extract the inputs
    email = parsed_inputs.get('email', "test@gmail.com")
    order_result = parsed_inputs.get('order_result', OrderResult("123", "123", "123", "1234", {"123": 123}))
    # carrier = parsed_inputs.get('carrier', {})

    # 3. Call the function
    send_email_confirmation(email, order_result)
