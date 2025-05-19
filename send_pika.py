import pika
from ia_utils.settings import Settings
from uuid import uuid4
import time
import orjson
settings = Settings()

# TODO universalize this


EXCHANGES = settings.AMQP.EXCHANGES
ROUTING = settings.AMQP.ROUTING

url = pika.URLParameters(settings.AIR_RABBITMQ_BROKER)
connection = pika.BlockingConnection(url)
channel = connection.channel()


props = pika.BasicProperties(
    expiration=str(5 * 60 * 1000),
    content_type='application/json')

def send_message(**kwargs):

    msg = {
            'timestamp': int(time.time()),
            'version': settings.API_VERSION,
            'data': kwargs
        }
    body = orjson.dumps(kwargs)

    channel.basic_publish(
        exchange=EXCHANGES.FLOW,
        routing_key=ROUTING.EVENTS_CREATED,
        properties=props,
        body=body
    )


send_message(
    aircraft_type='320A',
    cctv_id='PMI_56_R',
    gate_id='PMI_56',
    turnaround_id=str(uuid4()),
    processed_frame_ts=1741333043,
    events_frame_ts=1741333043,
    events=[{
        'event': 'arrival-prep:aircraft-stopped:aircraft-stopped',
        'track_id': 19,
        'confidence': 99.3,
        'normalized_bbox': normalize_bbox((0, 0, 100, 100), (1920, 1080)),
    }]
)
