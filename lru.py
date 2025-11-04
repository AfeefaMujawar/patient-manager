pages = list(map(int, input("Enter pages: ").split()))
cap = int(input("Enter frame capacity: "))

frames = []
use = {}        # track last used index
faults = hits = 0

for i, p in enumerate(pages):
    if p in frames:
        hits += 1
    else:
        faults += 1
        if len(frames) < cap:
            frames.append(p)
        else:
            # find least recently used page
            lru = min(frames, key=lambda x: use[x])
            frames[frames.index(lru)] = p
    use[p] = i   # update last used index
    print(*frames)

print("Total Page Faults:", faults)
print("Total Page Hits:", hits)