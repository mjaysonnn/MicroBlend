# For Jaeger
import requests
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# get ec2 ip address
r = requests.get('http://169.254.169.254/latest/meta-data/local-ipv4')
ec2_ip_address = r.text
jaeger_endpoint = f"http://{ec2_ip_address}:4318/v1/traces"


def set_tracer():
    """
    Set up the tracer
    """
    trace.set_tracer_provider(TracerProvider())  # set the tracer provider
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(OTLPSpanExporter(jaeger_endpoint)))
    tracer = trace.get_tracer(__name__)  # get the tracer

    return tracer
