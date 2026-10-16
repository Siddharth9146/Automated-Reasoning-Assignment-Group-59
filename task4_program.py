# Task 4 - Program Verification

"""
Consider the following program written in C. Here, __nondet__() is a function that
nondeterministically returns an arbitrary Boolean value.

int main() {
    int a = 1;
    int b = 1;
    while (a < 180) {
        if (__nondet__()) {
            b = b + 3;
            a = a + 2*b;
        } else {
            b = b + a;
            a = a + 5;
        }
    }
    if (b == 190 + k) {
        assert(false);
    }
}

Show for each value of k in {0, ..., 10} whether the program is safe or can crash (i.e. reach
the assert statement). If a crash is possible, provide one complete run leading to that crash.
"""

from z3 import *

# a grows by at least 5 every iteration, so the loop runs at most 36 times
max_steps = 36

a = [Int(f"a_{i}") for i in range(max_steps + 1)]
b = [Int(f"b_{i}") for i in range(max_steps + 1)]
choice = [Bool(f"choice_{i}") for i in range(max_steps)]

s = Solver()
s.add(a[0] == 1, b[0] == 1)

for i in range(max_steps):
    then_branch = And(b[i + 1] == b[i] + 3, a[i + 1] == a[i] + 2 * b[i + 1])
    else_branch = And(b[i + 1] == b[i] + a[i], a[i + 1] == a[i] + 5)
    step = If(choice[i], then_branch, else_branch)

    # once the loop is done the values stay the same
    s.add(If(a[i] < 180, step, And(a[i + 1] == a[i], b[i + 1] == b[i])))

# the loop must have ended
s.add(a[max_steps] >= 180)

for k in range(11):
    s.push()
    s.add(b[max_steps] == 190 + k)
    if s.check() == sat:
        m = s.model()
        run = [f"(a={m.evaluate(a[0])}, b={m.evaluate(b[0])})"]
        for i in range(max_steps):
            if m.evaluate(a[i]).as_long() >= 180:
                break
            branch = "then" if is_true(m.evaluate(choice[i])) else "else"
            run.append(f"{branch} -> (a={m.evaluate(a[i + 1])}, b={m.evaluate(b[i + 1])})")
        print(f"k = {k}: can crash")
        print("  " + "\n  ".join(run))
    else:
        print(f"k = {k}: safe")
    s.pop()
