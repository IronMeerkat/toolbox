import threading
import queue
from time import sleep

q = queue.Queue()

def worker():
    while True:
        item = q.get()
        print(f'Working on {item}')
        sleep(3)
        print(f'Finished {item}')
        q.task_done()

# Turn-on the worker thread.
threading.Thread(target=worker, daemon=True).start()


# Send thirty task requests to the worker.
for item in range(5):
    q.put(item)

sleep(10)


# Block until all tasks are done.
# q.join()
print('All work completed')
