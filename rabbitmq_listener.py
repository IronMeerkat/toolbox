import json
import pika

# TODO implement argparse
RABBITMQ_URL = ''
EXCHANGE = 'basic'
QUEUE = 'listener'
ROUTING_KEY = 'events'


all_data = []

# TODO make this a class
def on_message(channel, deliver, props, body):

    '''
    This function is called every time a message has been
    received.
    '''

    global all_data
    try:
        data = json.loads(body.decode())
        print(data)
        all_data.append(data)
    except UnicodeDecodeError as e:
        print(e)
    except json.JSONDecodeError as e:
        print(e.msg)
    except Exception as e:
        print(e)


parameters = pika.URLParameters(RABBITMQ_URL)
connection = pika.BlockingConnection(parameters=parameters)
channel = connection.channel()
channel.basic_qos(prefetch_count=1)
channel.queue_declare(QUEUE)

channel.queue_bind(QUEUE, EXCHANGE, routing_key=ROUTING_KEY)
channel.basic_consume(QUEUE, on_message, auto_ack=True)

try:
    channel.start_consuming()
except KeyboardInterrupt:
    exit()
except Exception:
    channel.stop_consuming()

connection.close()
