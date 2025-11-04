class Process:
    def __init__(self, pid, at, bt):
        self.pid = pid
        self.at = at
        self.bt = bt
        self.rt = bt
        self.ct = self.tat = self.wt = 0

def srtf(processes):
    time = completed = 0
    n = len(processes)
    while completed < n:
        ready = [p for p in processes if p.at <= time and p.rt > 0]
        if ready:
            p = min(ready, key=lambda x: x.rt)
            p.rt -= 1
            time += 1
            if p.rt == 0:
                p.ct = time
                p.tat = p.ct - p.at
                p.wt = p.tat - p.bt
                completed += 1
        else:
            time += 1

def print_results(processes):
    print("\nPID\tAT\tBT\tCT\tTAT\tWT")
    total_wt = total_tat = 0
    for p in processes:
        total_wt += p.wt
        total_tat += p.tat
        print(f"{p.pid}\t{p.at}\t{p.bt}\t{p.ct}\t{p.tat}\t{p.wt}")
    print(f"\nAverage Waiting Time: {total_wt/len(processes):.2f}")
    print(f"Average Turnaround Time: {total_tat/len(processes):.2f}")

n = int(input("Enter number of processes: "))
processes = []
for i in range(n):
    pid = input(f"Enter Process ID for P{i+1}: ")
    at = int(input(f"Enter Arrival Time for {pid}: "))
    bt = int(input(f"Enter Burst Time for {pid}: "))
    processes.append(Process(pid, at, bt))

processes.sort(key=lambda x: x.at)
srtf(processes)
print_results(processes)
