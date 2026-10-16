# Task 1 - Magic Factory

"""
Six trucks (with rated speed 80km/h) have to deliver obscure building blocks to a magic
factory. The distance to the factory is 120 km. Every truck has a max load of 8000 kg and
can fit at most ten building blocks. In total, the following has to be delivered:
• Six nuzzles, each of weight 800 kg.
• Twelve prittles, each of weight 405 kg.
• Fifteen skipples, each of weight 500 kg.
• Eight crottles, each of weight 2500 kg.
• A number of dupples, each of weight 600 kg.
If more than two types of goods are present in a truck, it has to drive a a reduced speed
of 60 km/h to lower the risk of magic interactions between the goods. Skipples are round and
elastic, only two out of the six trucks have facilities for fastening them. Prittles are somewhat
valuable: those pallets need to be distributed over at least five trucks.
(a) Investigate what is the maximum number of dupples that can be delivered in a single
run, and show how for that number all goods may be divided over the six trucks.
(b) Do the same, with the extra information that crottles get too cold if there are less than
two dupples in the same truck
"""

from z3 import *

trucks = []
for i in range(6):
    trucks.append({
        "nuzzles": Int(f"t{i}_nuzzles"),
        "prittles": Int(f"t{i}_prittles"),
        "skipples": Int(f"t{i}_skipples"),
        "crottles": Int(f"t{i}_crottles"),
        "dupples": Int(f"t{i}_dupples")
    })

weights = {
    "nuzzles": 800,
    "prittles": 405,
    "skipples": 500,
    "crottles": 2500,
    "dupples": 600
}

solver = Optimize()

# truck limits
for t in trucks:
    for name in weights:
        solver.add(t[name] >= 0)
    
    # max 10 items total per truck
    solver.add(Sum(list(t.values())) <= 10)
    
    # max 8000 kg capacity per truck
    truck_weight = Sum([t[name] * weights[name] for name in weights])
    solver.add(truck_weight <= 8000)

# global delivery requirements
solver.add(Sum([t["nuzzles"] for t in trucks]) == 6)
solver.add(Sum([t["prittles"] for t in trucks]) == 12)
solver.add(Sum([t["skipples"] for t in trucks]) == 15)
solver.add(Sum([t["crottles"] for t in trucks]) == 8)

# skipples: only 2 trucks can fasten them, so restrict to the first two
for t in trucks[2:]:
    solver.add(t["skipples"] == 0)

# prittles: distribute over at least 5 trucks
prittle_trucks = Sum([If(t["prittles"] > 0, 1, 0) for t in trucks])
solver.add(prittle_trucks >= 5)

# speed rule (not relevant)
# No use in the solver, but we could use it to calculate the total time ig
speed_reduction = []
for t in trucks:
    num_types = Sum([If(t[name] > 0, 1, 0) for name in weights])
    speed_reduction.append(If(num_types > 2, 1, 0))

# maximize the dupples
total_dupples = Sum([t["dupples"] for t in trucks])
solver.maximize(total_dupples)

print("\nPart A")
if solver.check() == sat:
    m = solver.model()
    print("Max dupples:", m.evaluate(total_dupples))
    for i, t in enumerate(trucks):
        goods = [f"{m[t[name]].as_long()} {name}" for name in weights]
        print(f"Truck {i + 1}: {', '.join(goods)}")
else:
    print("No solution found.")

# part b: add the extra constraint for cold crottles
for t in trucks:
    solver.add(Implies(t["crottles"] > 0, t["dupples"] >= 2))

print("\nPart B ")
if solver.check() == sat:
    m = solver.model()
    print("Max dupples:", m.evaluate(total_dupples))
    for i, t in enumerate(trucks):
        goods = [f"{m[t[name]].as_long()} {name}" for name in weights]
        print(f"Truck {i + 1}: {', '.join(goods)}")
else:
    print("No solution found.")