import csv
import time
from datetime import datetime
from kubernetes import client, config
import sys

# Load kube config
config.load_kube_config(config_file='~/.kube/config')

# TODO implement argparse
namespace = sys.argv[1]
csv_file_path = sys.argv[2] or 'pod_metrics.csv'


def get_pod_metrics(api_instance, namespace):
    """Fetches CPU and Memory usage for all pods in the given namespace using the Metrics API."""
    metrics_data = []
    current_time = datetime.now().isoformat()

    # Accessing metrics using CustomObjectsApi
    try:
        pod_metrics = api_instance.get_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=namespace,
            plural="pods",
            name=""
        )
        # TODO implement threading
        for pod in pod_metrics['items']:
            pod_name = pod['metadata']['name']
            for container in pod['containers']:
                cpu_usage = container['usage']['cpu']
                memory_usage = container['usage']['memory']
                metrics_data.append(
                    [current_time, pod_name, container['name'], cpu_usage, memory_usage])
    except Exception as e:
        print(f"Error fetching pod metrics: {e}")

    return metrics_data


def append_to_csv(file_path, data):
    """Appends the given data to a CSV file."""
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)


# Write headers to the CSV if it's the first run
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "pod", "deployment",
                    "cpu", "ram"])

# Instantiate the API
api_instance = client.CustomObjectsApi()

while True:
    pod_metrics = get_pod_metrics(api_instance, namespace)
    if pod_metrics:
        append_to_csv(csv_file_path, pod_metrics)
    time.sleep(1)  # Wait for 1 second before the next query
