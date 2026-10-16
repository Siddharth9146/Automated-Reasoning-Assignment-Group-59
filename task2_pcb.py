# Task 2 - PCB layout

"""
Modern foldable phones contain multiple printed circuit boards (PCBs), usually one on each
side of the hinge. A phone manufacturer needs to design the internal layout for their next
flagship model. And so, they have to fit some chips on these PCBs. What makes it challenging
is that the PCBs are odd-shaped in order to fit tightly around the battery and camera. The
design should meet the following constraints:
• There are 14 chips that need to fit on the PCBs, each chip is placed exactly once. The
dimensions of the PCBs and the size of each chip is given in Figure 1.
• Three of the chips can become warm under load (see again Figure 1). To prevent
overheating, their centres should not be closer than 20 length units in either the x or y
direction.
• Chips may be rotated 90◦
, but may not overlap.
(a) Give a layout of the PCBs following the above constraints.
(b) To boost the spec sheet, the camera department would like to integrate more cameras.
This will increase the width of the camera cutout by 4 length units for each additional
camera. What happens if one camera is added?
(c) And what if three cameras are added?

Rest of the diagrams and figures in the assignment PDF, too lazy to copy them here
"""

from z3 import *

n_chips = 14
chip_specs = [
    (5, 6, True),
    (5, 6, True),
    (4, 6, False),
    (4, 10, False),
    (5, 7, False),
    (3, 7, True),
    (7, 7, False),
    (6, 10, False),
    (6, 12, False),
    (4, 10, False),
    (6, 9, False),
    (5, 11, False),
    (6, 10, False),
    (5, 10, False)
]

# PCB dimensions
pcb1_specs = {
    "width": 20,
    "height": 20
    }

pcb1_battery_cutout = {
    "x": 0,
    "y": 0,
    "width": 12,
    "height": 8
}

pcb2_specs = {
    "width": 22,
    "height": 27
}

pcb2_battery_cutout = {
    "x": 0,
    "y": 0,
    "width": 14,
    "height": 12
}


def outside(xi, yi, wi, hi, cut):
    return Or(xi + wi <= cut["x"],
              xi >= cut["x"] + cut["width"],
              yi + hi <= cut["y"],
              yi >= cut["y"] + cut["height"])


def solve(camera_width):
    # camera cutout grows to the left from the top right corner
    pcb2_camera_cutout = {
        "x": pcb2_specs["width"] - camera_width,
        "y": pcb2_specs["height"] - 5,
        "width": camera_width,
        "height": 5,
    }

    # Variables for chip positions and orientations
    x = [Int(f"x_{i}") for i in range(n_chips)]
    y = [Int(f"y_{i}") for i in range(n_chips)]
    rotated = [Bool(f"rotated_{i}") for i in range(n_chips)]
    what_pcb = [Int(f"pcb_{i}") for i in range(n_chips)]

    # Rotation changes
    w = [If(rotated[i], chip_specs[i][1], chip_specs[i][0]) for i in range(n_chips)]
    h = [If(rotated[i], chip_specs[i][0], chip_specs[i][1]) for i in range(n_chips)]

    s = Solver()

    # PCB assignment constraints
    for i in range(n_chips):
        s.add(Or(what_pcb[i] == 0, what_pcb[i] == 1))
        s.add(x[i] >= 0)
        s.add(y[i] >= 0)

    # PCB boundary and cutout constraints
    for i in range(n_chips):
        s.add(If(what_pcb[i] == 0,
                 And(x[i] + w[i] <= pcb1_specs["width"],
                     y[i] + h[i] <= pcb1_specs["height"],
                     outside(x[i], y[i], w[i], h[i], pcb1_battery_cutout)),
                 And(x[i] + w[i] <= pcb2_specs["width"],
                     y[i] + h[i] <= pcb2_specs["height"],
                     outside(x[i], y[i], w[i], h[i], pcb2_battery_cutout),
                     outside(x[i], y[i], w[i], h[i], pcb2_camera_cutout))))

    # Non-overlapping constraints (only on the same PCB)
    for i in range(n_chips):
        for j in range(i + 1, n_chips):
            other = {"x": x[j], "y": y[j], "width": w[j], "height": h[j]}
            s.add(Implies(what_pcb[i] == what_pcb[j],
                          outside(x[i], y[i], w[i], h[i], other)))

    # Overheating constraints (centres doubled to stay in integers)
    hot = [i for i in range(n_chips) if chip_specs[i][2]]
    for a in range(len(hot)):
        for b in range(a + 1, len(hot)):
            i, j = hot[a], hot[b]
            cx_i, cy_i = 2 * x[i] + w[i], 2 * y[i] + h[i]
            cx_j, cy_j = 2 * x[j] + w[j], 2 * y[j] + h[j]
            s.add(Implies(what_pcb[i] == what_pcb[j],
                          Or(Abs(cx_i - cx_j) >= 40, Abs(cy_i - cy_j) >= 40)))

    # Solve
    if s.check() == sat:
        m = s.model()
        for p in range(2):
            print(f"PCB {p + 1}:")
            for i in range(n_chips):
                if m.evaluate(what_pcb[i]) == p:
                    xi = m.evaluate(x[i])
                    yi = m.evaluate(y[i])
                    wi = m.evaluate(w[i])
                    hi = m.evaluate(h[i])
                    rot = "rotated" if is_true(m.evaluate(rotated[i])) else ""
                    heat = "hot" if chip_specs[i][2] else ""
                    print(f"  Chip {i + 1}: x={xi}, y={yi}, w={wi}, h={hi} {rot} {heat}")
    else:
        print("No solution found.")


print("\nPart A")
solve(7)

print("\nPart B")
solve(7 + 4)

print("\nPart C")
solve(7 + 3 * 4)
