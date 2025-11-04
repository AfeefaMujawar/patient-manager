import threading, time
from queue import Queue

q = Queue(maxsize=3)

def producer():
    for i in range(5):
        if q.full():
            print("List is full, producer will wait")
        q.put(i)
        print("Produced", i)
        if q.qsize() == 1:
            print("Space in queue, Consumer notified the producer")
        time.sleep(0.1)
    q.put(None)  # stop signal

def consumer():
    while True:
        item = q.get()
        if item is None:
            break
        print("Consumed", item)
        q.task_done()
        time.sleep(0.5)

t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer)
t1.start()
t2.start()
t1.join()
t2.join()