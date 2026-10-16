# Task 3 - Food Supply

"""
Three non-self-supporting villages A, B and C consume one food package each per time unit.
The required food packages are delivered by a truck, having a capacity of 130 food packages.
The truck has to pick up its food packages at location S containing an unbounded supply.
Here every number indicates the number of time units the truck
needs to travel from one village to another, including loading
and delivering. There is often a lot of traffic going from B to C,
hence it takes a bit longer than the journey from C to B. The
villages only have a limited capacity to store food packages: for
A and C this capacity is 90, for B it is 120. Initially, the truck
is in S and is fully loaded, and in each of the three villages there
are 60 food packages.
(a) Show that it is impossible to deliver food packages in such a way that each of the villages
consumes one food package per time unit forever. For how long can the villages be kept
alive at best?
(b) Show that this is possible if the capacity of the truck is increased to 150 food packages.
What is the shortest possible driving schedule for the truck to sustain the villages
indefinitely?
"""

from z3 import *

S, A, B, C = 0, 1, 2, 3
names = ["S", "A", "B", "C"]
villages = [A, B, C]

roads = [
    (S, A, 15), 
    (A, S, 15),
    (S, C, 15),
    (C, S, 15),
    (A, C, 12),
    (C, A, 12),
    (A, B, 17),
    (B, A, 17),
    (C, B, 9),
    (B, C, 13)
]

village_cap = {A: 90, B: 120, C: 90}
start_stock = 60 


def make_steps(n):
    loc = [Int(f"loc_{k}") for k in range(n + 1)]
    load = [Int(f"load_{k}") for k in range(n + 1)]
    stock = [{v: Int(f"stock_{names[v]}_{k}") for v in villages} for k in range(n + 1)]
    deliver = [Int(f"deliver_{k}") for k in range(n + 1)]
    t = [Int(f"t_{k}") for k in range(n + 1)]
    return loc, load, stock, deliver, t


def add_run(s, n, truck_cap, steps):
    loc, load, stock, deliver, t = steps

    # initial state
    s.add(loc[0] == S, load[0] == truck_cap, t[0] == 0)
    for v in villages:
        s.add(stock[0][v] == start_stock)

    for k in range(n):
        s.add(deliver[k + 1] >= 0)
        options = []
        for (u, v, d) in roads:
            step = [loc[k] == u, loc[k + 1] == v, t[k + 1] == t[k] + d]

            # every village must survive the trip
            for x in villages:
                step.append(stock[k][x] - d >= 0)

            if v == S:
                step.append(load[k + 1] == truck_cap)
                step.append(deliver[k + 1] == 0)
                for x in villages:
                    step.append(stock[k + 1][x] == stock[k][x] - d)
            else:
                step.append(deliver[k + 1] <= load[k])
                step.append(load[k + 1] == load[k] - deliver[k + 1])
                for x in villages:
                    if x == v:
                        step.append(stock[k + 1][x] == stock[k][x] - d + deliver[k + 1])
                        step.append(stock[k + 1][x] <= village_cap[x])
                    else:
                        step.append(stock[k + 1][x] == stock[k][x] - d)
            options.append(And(step))
        s.add(Or(options))


def print_run(m, steps, upto):
    loc, load, stock, deliver, t = steps
    for k in range(upto + 1):
        where = names[m.evaluate(loc[k]).as_long()]
        stocks = ", ".join(f"{names[v]}={m.evaluate(stock[k][v])}" for v in villages)
        print(f"  t={m.evaluate(t[k])}: at {where}, delivered {m.evaluate(deliver[k])}, "
              f"truck load {m.evaluate(load[k])}, {stocks}")


# Part A
print("\nPart A")
best_time = 0
best = None
n = 1
while True:
    s = Optimize()
    steps = make_steps(n)
    add_run(s, n, 130, steps)

    # villages live until the last arrival plus the smallest remaining stock
    loc, load, stock, deliver, t = steps
    last = stock[n]
    min_stock = If(last[A] <= last[B],
                   If(last[A] <= last[C], last[A], last[C]),
                   If(last[B] <= last[C], last[B], last[C]))
    alive = t[n] + min_stock
    s.maximize(alive)

    if s.check() != sat:
        print(f"No run of {n} steps exists, so the villages cannot be sustained forever")
        break

    m = s.model()
    value = m.evaluate(alive).as_long()
    if value > best_time:
        best_time = value
        best = (m, steps, n)
    n += 1

print("Villages can be kept alive for at most", best_time, "time units")
m, steps, k = best
print_run(m, steps, k)


# Part B
print("\nPart B")
n = 1
while True:
    s = Optimize()
    steps = make_steps(n)
    add_run(s, n, 150, steps)
    loc, load, stock, deliver, t = steps

    # the last state is at least as good as some earlier state, so the loop can repeat forever
    loops = []
    for i in range(n):
        same = [loc[n] == loc[i], load[n] >= load[i]]
        for v in villages:
            same.append(stock[n][v] >= stock[i][v])
        loops.append(And(same))
    s.add(Or(loops))

    if s.check() == sat:
        start = Int("loop_start")
        s.add(start >= 0, start < n)
        for i in range(n):
            s.add(Implies(start == i, loops[i]))
        loop_time = Sum([If(start == i, t[n] - t[i], 0) for i in range(n)])
        s.minimize(loop_time)
        s.check()
        m = s.model()
        i = m.evaluate(start).as_long()
        print(f"Found a schedule with {n} steps, repeating from step {i} "
              f"(loop takes {m.evaluate(loop_time)} time units)")
        print_run(m, steps, n)
        break
    n += 1
