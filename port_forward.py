import portforward
import os
import pymongo

# TODO implement for multiple types of services
class PortForward:

    def __init__(self, namespace, pod_name='mongodb-rs-0', local_port=27018, pod_port=27017):
        self.namespace = namespace
        self.pod_name = pod_name
        self.local_port = local_port
        self.pod_port = pod_port

    def __enter__(self):
        portforward.__enter__(self.namespace, self.pod_name, self.local_port, self.pod_port)

    def __exit__(*exc_info):
        portforward.__exit__(*exc_info)



