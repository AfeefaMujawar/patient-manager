def rr(processes, q):
    processes.sort(key=lambda p: p['at'])
    n = len(processes)
    time = 0; i = 0; done = 0; queue = []; order = []
    # init fields
    for p in processes:
        p.update({'rem': p['bt'], 'start': -1, 'ct':0, 'tat':0, 'wt':0, 'rt':0})
    while done < n:
        # enqueue arrivals
        while i < n and processes[i]['at'] <= time:
            queue.append(processes[i]); i += 1
        if not queue:
            time = processes[i]['at']   # jump to next arrival
            continue
        cur = queue.pop(0)
        if cur['start'] == -1:                  # first time scheduled
            cur['start'] = time
            cur['rt'] = cur['start'] - cur['at']
        run = min(cur['rem'], q)
        cur['rem'] -= run
        order.append(str(cur['pid']))
        time += run
        # enqueue arrivals that came during execution
        while i < n and processes[i]['at'] <= time:
            queue.append(processes[i]); i += 1
        if cur['rem'] == 0:
            cur['ct'] = time
            cur['tat'] = cur['ct'] - cur['at']
            cur['wt'] = cur['tat'] - cur['bt']
            done += 1
        else:
            queue.append(cur)
    return order

def print_table(processes, order):
    print("\nPID  AT  BT  CT  TAT  WT  RT")
    sw = st = sr = 0
    for p in processes:
        print(f"{p['pid']:<5}{p['at']:<5}{p['bt']:<5}{p['ct']:<5}{p['tat']:<5}{p['wt']:<5}{p['rt']:<5}")
        sw += p['wt']; st += p['tat']; sr += p['rt']
    n = len(processes)
    print(f"\nAverage Waiting Time: {sw/n:.2f}")
    print(f"Average Turnaround Time: {st/n:.2f}")
    print(f"Average Response Time: {sr/n:.2f}")
    print("\nExecution Order (Queue Timeline): " + " -> ".join(order))

if __name__ == "__main__":
    n = int(input("Enter number of processes: "))
    procs = []
    for i in range(n):
        pid = int(input(f"Enter Process ID for P{i+1}: "))
        at  = int(input(f"Enter Arrival Time for {pid}: "))
        bt  = int(input(f"Enter Burst Time for {pid}: "))
        procs.append({'pid':pid, 'at':at, 'bt':bt})
    q = int(input("Enter Time Quantum: "))
    # keep a copy of original input order for printing (optional)
    orig = procs.copy()
    order = rr(procs, q)
    # print in original input order to match screenshot style
    print_table(orig, order)