# Task 5 - Configurable Systems

import sys
import time
from oxidd.bdd import BDDManager

INF = float("inf")


def read_file(path):
    """Returns n features, clauses and the variable order."""
    n = 0
    clauses = []
    var_order = None
    constraint = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "c":
                if len(parts) > 1 and parts[1] == "vo":
                    var_order = []
                    for t in parts[2:]:
                        if t == "0":
                            break
                        var_order.append(int(t))
                continue
            if parts[0] == "p":
                n = int(parts[2])
                continue
            for tok in parts:
                lit = int(tok)
                if lit == 0:
                    clauses.append(constraint)
                    constraint = []
                else:
                    constraint.append(lit)
    if constraint:
        clauses.append(constraint)
    var_order = var_order or list(range(1, n + 1))
    seen = set(var_order)
    for v in range(1, n + 1):
        if v not in seen:
            var_order.append(v)  # missing ones go last

    return n, clauses, var_order


def cheapest_path(mgr, u):
    """Fewest features set to 1 in a satisfying assignment of u (INF if unsatisfiable)."""
    memo = {mgr.false(): INF, mgr.true(): 0}
    stack = [u]

    while len(stack) > 0:  # iterative post-order traversal
        f = stack[-1]
        if f in memo:
            stack.pop()
            continue

        high, low = f.cofactors()

        if low in memo and high in memo:
            memo[f] = min(memo[low], 1 + memo[high])
            stack.pop()
        else:
            if low not in memo:
                stack.append(low)
            if high not in memo:
                stack.append(high)

    return memo[u]


def exact_k_features(mgr, vars_in_order, k):
    """BDD of 'exactly k of these variables are true'."""
    nxt = [mgr.true()] + [mgr.false()] * k

    for v in reversed(vars_in_order):
        x = mgr.var(v)
        constraint = [x.ite(mgr.false(), nxt[0])]
        for c in range(1, k + 1):
            constraint.append(x.ite(nxt[c - 1], nxt[c]))
        nxt = constraint

    return nxt[k]


def analyze(path, essential_features=(218, 311)):
    t0 = time.time()

    n, clauses, var_order = read_file(path)

    mgr = BDDManager(100_000_000, 1_000_000, 1)
    mgr.add_vars(n)

    var_of = {}
    for i, feature in enumerate(var_order):
        var_of[feature] = i

    def lit(l):
        if l > 0:
            return mgr.var(var_of[l])
        else:
            return ~mgr.var(var_of[-l])

    # sorting clauses for better BDD building performance
    clauses = sorted(clauses, key=lambda clause: -min(var_of[abs(l)] for l in clause))

    phi = mgr.true()
    for clause in clauses:
        c = mgr.false()
        for l in clause:
            c = c | lit(l)
        phi = phi & c

    res = {}
    res["file"] = path
    res["vars"] = n
    res["clauses"] = len(clauses)

    res["i_nodes"] = phi.node_count()
    res["ii_valid"] = phi.sat_count(n)  # exact (arbitrary-precision) count

    k = cheapest_path(mgr, phi)
    res["iii_min"] = k

    valid_features = True
    for e in essential_features:
        if not (1 <= e <= n):
            valid_features = False

    if valid_features:
        phis = phi
        for e in essential_features:
            phis = phis & mgr.var(var_of[e])
        res["iv_min_S"] = cheapest_path(mgr, phis)
    else:
        res["iv_min_S"] = "n/a (feature does not exist)"

    if k != INF:  # (v), (vi): phi /\ exactly_k
        all_min = phi & exact_k_features(mgr, list(range(n)), k)
        res["v_nodes"] = all_min.node_count()
        res["vi_count"] = all_min.sat_count(n)

    res["seconds"] = round(time.time() - t0, 2)
    return res


# set max string digits just in case we have massive numbers
sys.set_int_max_str_digits(0)

if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = analyze(p)
        print(
            f"{r['file']} ({r['vars']} features, {r['clauses']} clauses, {r['seconds']} s)"
        )
        print(f"  (i)   nodes of phi        : {r['i_nodes']}")
        print(f"  (ii)  valid configurations: {r['ii_valid']}")
        print(f"  (iii) minimum size        : {r['iii_min']}")
        print(f"  (iv)  minimum size with S : {r['iv_min_S']}")
        if "v_nodes" in r:
            print(f"  (v)   nodes all-minimum   : {r['v_nodes']}")
            print(f"  (vi)  minimum configs     : {r['vi_count']}")
