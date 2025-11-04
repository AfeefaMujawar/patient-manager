P = int(input("Enter number of processes: "))
R = int(input("Enter number of resources: "))

maxm, alloc = [], []
print("Enter Max matrix:")
for i in range(P):
    maxm.append(list(map(int, input().split())))

print("Enter Allocation matrix:")
for i in range(P):
    alloc.append(list(map(int, input().split())))

avail = list(map(int, input("Enter Available resources: ").split()))

def is_safe(P, R, maxm, alloc, avail):
    need = [[maxm[i][j] - alloc[i][j] for j in range(R)] for i in range(P)]
    finish = [0]*P
    safe_seq, work = [], avail[:]

    count = 0
    while count < P:
        found = False
        for p in range(P):
            if not finish[p] and all(need[p][j] <= work[j] for j in range(R)):
                for k in range(R):
                    work[k] += alloc[p][k]
                safe_seq.append(p)
                finish[p], found, count = 1, True, count+1
        if not found:
            print("System is not in safe state")
            return
    print("System is in safe state.")
    print("Safe sequence:", safe_seq)

is_safe(P, R, maxm, alloc, avail)
