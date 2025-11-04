import sys

def sort(arr):
    arr.sort()

def fcfs(req, head):
    total = 0
    pos = head
    print("\nFCFS Disk Scheduling:")
    print("Sequence:", head, end="")
    for r in req:
        total += abs(r - pos)
        pos = r
        print(" ->", pos, end="")
    print("\nTotal Head Movement =", total)
    return total

def scan(req, head, size):
    temp = req[:]
    sort(temp)
    print("SCAN Disk Scheduling:")
    print("Sequence:", head, end="")
    total = 0
    idx = 0
    while idx < len(temp) and temp[idx] < head:
        idx += 1
    h = head
    for i in range(idx, len(temp)):
        total += abs(h - temp[i])
        h = temp[i]
        print(" ->", h, end="")
    if h != size - 1:
        total += abs((size - 1) - h)
        h = size - 1
        print(" ->", h, end="")
    for i in range(idx - 1, -1, -1):
        total += abs(h - temp[i])
        h = temp[i]
        print(" ->", h, end="")
    print("\nTotal Head Movement =", total)
    return total

def cscan(req, head, size):
    temp = req[:]
    sort(temp)
    print("C-SCAN Disk Scheduling:")
    print("Sequence:", head, end="")
    total = 0
    idx = 0
    while idx < len(temp) and temp[idx] < head:
        idx += 1
    h = head
    for i in range(idx, len(temp)):
        total += abs(h - temp[i])
        h = temp[i]
        print(" ->", h, end="")
    if h != size - 1:
        total += abs((size - 1) - h)
        h = size - 1
        print(" ->", h, end="")
    total += abs(h - 0)
    h = 0
    print(" ->", h, end="")
    for i in range(idx):
        total += abs(h - temp[i])
        h = temp[i]
        print(" ->", h, end="")
    print("\nTotal Head Movement =", total)
    return total

def look(req, head):
    temp = req[:]
    sort(temp)
    print("LOOK Disk Scheduling:")
    print("Sequence:", head, end="")
    total = 0
    idx = 0
    while idx < len(temp) and temp[idx] < head:
        idx += 1
    h = head
    for i in range(idx, len(temp)):
        total += abs(h - temp[i])
        h = temp[i]
        print(" ->", h, end="")
    for i in range(idx - 1, -1, -1):
        total += abs(h - temp[i])
        h = temp[i]
        print(" ->", h, end="")
    print("\nTotal Head Movement =", total)
    return total

def clook(req, head):
    temp = req[:]
    sort(temp)
    print("C-LOOK Disk Scheduling:")
    print("Sequence:", head, end="")
    total = 0
    idx = 0
    while idx < len(temp) and temp[idx] < head:
        idx += 1
    h = head
    for i in range(idx, len(temp)):
        total += abs(h - temp[i])
        h = temp[i]
        print(" ->", h, end="")
    total += abs(h - temp[0])
    h = temp[0]
    print(" ->", h, end="")
    for i in range(1, idx):
        total += abs(h - temp[i])
        h = temp[i]
        print(" ->", h, end="")
    print("\nTotal Head Movement =", total)
    return total

def sstf(req, head):
    print("SSTF Disk Scheduling:")
    print("Sequence:", head, end="")
    requests = req[:]
    total = 0
    pos = head
    while requests:
        # find request with minimum seek time
        nearest = min(requests, key=lambda x: abs(x - pos))
        total += abs(nearest - pos)
        pos = nearest
        print(" ->", pos, end="")
        requests.remove(nearest)
    print("\nTotal Head Movement =", total)
    return total


# -------- MAIN --------
n = int(input("Enter number of disk requests: "))
req = list(map(int, input("Enter disk requests: ").split()))
head = int(input("Enter initial head position: "))
size = int(input("Enter total disk size (e.g. 200): "))

fcfs(req, head)
scan(req, head, size)
cscan(req, head, size)
look(req, head)
clook(req, head)
sstf(req, head)