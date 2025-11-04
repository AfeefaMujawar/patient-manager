def fcfs_scheduling(processes):
    processes.sort(key=lambda x: x['arrival_time'])
    current_time = 0
    for p in processes:
        if current_time < p['arrival_time']:
            current_time = p['arrival_time']
        p['start_time'] = current_time
        p['completion_time'] = current_time + p['burst_time']
        p['turnaround_time'] = p['completion_time'] - p['arrival_time']
        p['waiting_time'] = p['turnaround_time'] - p['burst_time']
        current_time += p['burst_time']
    return processes

def print_table(processes):
    print("\nPID  AT  BT  CT  TAT  WT")
    total_wt = total_tat = 0
    for p in processes:
        print(f"{p['pid']:<5}{p['arrival_time']:<5}{p['burst_time']:<5}"
              f"{p['completion_time']:<5}{p['turnaround_time']:<5}{p['waiting_time']:<5}")
        total_wt += p['waiting_time']
        total_tat += p['turnaround_time']
    n = len(processes)
    print(f"\nAverage Waiting Time: {total_wt/n:.2f}")
    print(f"Average Turnaround Time: {total_tat/n:.2f}")

if __name__ == "__main__":
    n = int(input("Enter number of processes: "))
    print("Enter PID, Arrival Time (AT), Burst Time (BT):")
    processes = []
    for i in range(n):
        pid = int(input(f"Process {i+1} PID: "))
        at = int(input(f"Arrival Time for PID {pid}: "))
        bt = int(input(f"Burst Time for PID {pid}: "))
        processes.append({'pid': pid, 'arrival_time': at, 'burst_time': bt})
    scheduled = fcfs_scheduling(processes)
    print_table(scheduled)
