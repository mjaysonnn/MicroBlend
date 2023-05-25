from pathlib import Path
from typing import Union

from fastapi import FastAPI

from utils import utils

# Init FastAPI Application
app = FastAPI()

# Logging to file
logger = utils.init_logger(Path(__file__).parent.absolute())

# Make a dictionary of key with USD Dollar and value as rate
currency_dict = {
    "EUR": "1.0",
    "USD": "1.1305",
    "JPY": "126.40",
    "BGN": "1.9558",
    "CZK": "25.592",
    "DKK": "7.4609",
    "GBP": "0.85970",
    "HUF": "315.51",
    "PLN": "4.2996",
    "RON": "4.7463",
    "SEK": "10.5375",
    "CHF": "1.1360",
    "ISK": "136.80",
    "NOK": "9.8040",
    "HRK": "7.4210",
    "RUB": "74.4208",
    "TRY": "6.1247",
    "AUD": "1.6072",
    "BRL": "4.2682",
    "CAD": "1.5128",
    "CNY": "7.5857",
    "HKD": "8.8743",
    "IDR": "15999.40",
    "ILS": "4.0875",
    "INR": "79.4320",
    "KRW": "1275.05",
    "MXN": "21.7999",
    "MYR": "4.6289",
    "NZD": "1.6679",
    "PHP": "59.083",
    "SGD": "1.5349",
    "THB": "36.012",
    "ZAR": "16.0583",
}


def change_to_another_currency(total_cost, output_currency):
    """
    Given the total cost of a product, this function will convert it to another currency.
    """
    output_rate = currency_dict[output_currency]
    return float(total_cost) * float(output_rate)


@app.get("/currency_service/{input_p}")
def run_currency_service(input_p: Union[str, None] = None):
    # 1. Decode the input
    parsed_inputs = utils.native_object_decoded(input_p) if input_p != "Test" else {}

    # 2. Extract the inputs
    total_cost = parsed_inputs.get("total_cost", 0)
    output_currency = parsed_inputs.get("output_currency", "USD")

    return change_to_another_currency(total_cost, output_currency)
