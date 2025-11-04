pages = list(map(int, input("Enter pages: ").split()))
cap = int(input("Enter frame capacity: "))

frames = []
ptr = 0                # FIFO pointer: next slot to replace
faults = hits = 0

for p in pages:
    if p in frames:
        hits += 1
        print(f"Page {p} :", *frames, "- hit")
    else:
        faults += 1
        if len(frames) < cap:
            frames.append(p)        # fill left to right
        else:
            frames[ptr] = p        # replace at pointer (in-place)
            ptr = (ptr + 1) % cap
        print(f"Page {p} :", *frames, "- fault")

print("Total Page Faults:", faults)
print("Total Page Hits:", hits)