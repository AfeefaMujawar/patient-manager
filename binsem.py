import threading, time

s = threading.Semaphore(1)

def critical_section(name):
    print(f"{name} is trying to enter the critical section...")
    s.acquire()
    print(f"{name} has entered the critical section.")
    time.sleep(2)
    print(f"{name} is leaving the critical section.")
    s.release()

t1 = threading.Thread(target=critical_section, args=("Thread 1",))
t2 = threading.Thread(target=critical_section, args=("Thread 2",))
t1.start(); t2.start()
t1.join(); t2.join()
print("Both threads have completed execution.")
