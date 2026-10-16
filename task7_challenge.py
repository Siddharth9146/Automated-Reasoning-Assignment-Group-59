# Task 7 - Challenge: 5G network planning

"""
A telecom company plans a 5G network for a city modelled as a 12x12 grid of cells.
Every population zone has a number of inhabitants. Towers can only be built on a
fixed list of candidate sites, and three tower types are available, each with its
own range and cost. A tower covers every cell within its range (Manhattan distance). At most one tower can be built on a site and the total cost may not
exceed the budget. Every tower gets one of k frequencies, and two towers 
on the same frequency may not be at distance <= r + r' (their coverage areas overlap).

(a) With the given budget and 3 frequencies, find the maximum population that can
    be covered, and among those a solution with the lowest cost.
(b) Find the minimum number of frequencies needed to keep the coverage of (a)
    within the same budget, and among those a solution with the lowest cost. Show
    that one frequency less is not enough.
"""

from z3 import *

budget = 15
n_frequencies = 3

# (name, range, cost)
tower_types = [
    ("macro", 3, 4),
    ("micro", 2, 3),
    ("small", 1, 2)
]

# candidate sites for towers
sites = [
    (3, 6), (5, 7), (6, 4), (8, 8), (7, 3), (4, 3),
    (2, 10), (4, 10), (10, 9), (8, 10),
    (2, 2), (3, 4), (9, 2), (11, 3)
]

# (x, y, population)
zones = [
    # downtown
    (4, 5, 90), (4, 7, 80), (6, 3, 100), (7, 6, 120), (5, 8, 70), (8, 6, 60),
    # north-west
    (1, 10, 40), (2, 9, 30), (3, 11, 25),
    # north-east
    (10, 10, 50), (9, 11, 35), (11, 8, 20),
    # south-west
    (1, 1, 30), (2, 3, 25),
    # south-east
    (10, 1, 40), (9, 3, 30)
]


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def build(s, k):
    cells = sites

    # p[c][t]: a tower of type t is built on site c
    p = {c: [Bool(f"p_{c[0]}_{c[1]}_{name}") for name, _, _ in tower_types] for c in cells}
    # f[c]: frequency of the tower on site c
    f = {c: Int(f"f_{c[0]}_{c[1]}") for c in cells}

    for c in cells:
        s.add(AtMost(*p[c], 1))
        s.add(f[c] >= 1, f[c] <= k)

    # budget
    cost = Sum([If(p[c][t], tower_types[t][2], 0) for c in cells for t in range(len(tower_types))])
    s.add(cost <= budget)

    # same frequency is not allowed when the coverage areas touch
    for i, c in enumerate(cells):
        for c2 in cells[i + 1:]:
            d = dist(c, c2)
            conflict = [And(p[c][a], p[c2][b])
                        for a in range(len(tower_types)) for b in range(len(tower_types))
                        if tower_types[a][1] + tower_types[b][1] >= d]
            if conflict:
                s.add(Implies(Or(conflict), f[c] != f[c2]))

    # zone z is covered if some tower has it within range
    covered = [Or([p[c][t] for c in cells for t in range(len(tower_types))
                   if dist(c, (zx, zy)) <= tower_types[t][1]])
               for zx, zy, _ in zones]
    population = Sum([If(covered[z], zones[z][2], 0) for z in range(len(zones))])

    return cells, p, f, covered, population, cost


def show(m, cells, p, f, covered, population, cost):
    total = sum(pop for _, _, pop in zones)
    print(f"Covered population: {m.evaluate(population)} of {total}")
    print(f"Cost: {m.evaluate(cost)} of {budget}")
    for c in cells:
        for t in range(len(tower_types)):
            if is_true(m.evaluate(p[c][t])):
                print(f"  Tower at {c}: {tower_types[t][0]}, frequency {m.evaluate(f[c])}")
    missed = [zones[z] for z in range(len(zones)) if not is_true(m.evaluate(covered[z]))]
    print("Uncovered zones:", ", ".join(f"({x},{y}):{pop}" for x, y, pop in missed) or "none")


print("\nPart A")
s = Optimize()
model = build(s, n_frequencies)
cells, p, f, covered, population, cost = model
# maximise covered population first, then minimise cost
for z in range(len(zones)):
    s.add_soft(covered[z], zones[z][2], id="coverage")
s.minimize(cost)
s.check()
m = s.model()
best = m.evaluate(population).as_long()
show(m, *model)

print("\nPart B")
for k in range(1, n_frequencies + 1):
    s = Optimize()
    model = build(s, k)
    s.add(model[4] >= best)
    s.minimize(model[5])
    result = s.check()
    print(f"k = {k}: {result}")
    if result == sat:
        show(s.model(), *model)
        break
