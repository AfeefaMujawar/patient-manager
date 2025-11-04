import threading, time

N = 5
chopstick = [threading.Lock() for _ in range(N)]

def philosopher(i):
    print(f"Philosopher {i+1} is thinking")
    with chopstick[i], chopstick[(i+1) % N]:
        print(f"Philosopher {i+1} is eating")
        time.sleep(1)
    print(f"Philosopher {i+1} finished eating")

threads = []
for i in range(N):
    t = threading.Thread(target=philosopher, args=(i,))
    threads.append(t)
    t.start()
    time.sleep(0.1)  # small delay like usleep(100000)

for t in threads:
    t.join()
