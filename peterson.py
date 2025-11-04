def peterson():
    flag = [False, False]
    turn = 0
    print("Peterson's Algorithm simulation for 2 processes")
    while True:
        p = int(input("Enter process number (0 or 1) wanting to enter critical section (or -1 to exit): "))
        if p == -1:
            print("Exiting simulation.")
            break
        if p not in [0, 1]:
            print("Invalid process number! Try again.")
            continue

        flag[p] = True
        turn = 1 - p
        print(f"Process {p} sets flag[{p}] = True and turn = {turn}")

        while flag[1 - p] and turn == 1 - p:
            pass

        print(f"Process {p} is entering the critical section.")
        input(f"Process {p} is in critical section. Press Enter to exit critical section...")

        flag[p] = False
        print(f"Process {p} exits critical section and sets flag[{p}] = False\n")

if __name__ == "__main__":
 peterson()