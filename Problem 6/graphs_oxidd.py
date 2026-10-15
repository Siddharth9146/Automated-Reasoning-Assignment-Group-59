import sys, time, math
import numpy as np
import pandas as pd
from oxidd.bdd import BDDManager
from oxidd.util import BooleanOperator


def read_table(path, cols=None):
    """Read a CSV/TSV file while allowing mixed text and numeric columns."""
    df = pd.read_csv(path, sep=None, engine="python")

    if cols is not None:
        df = df.iloc[:, list(cols)]

    return df


def build_set(mgr, keys, levels):
    """BDD of a set of integers.

    keys   : numpy array of distinct integers (each one is one element of the set)
    levels : levels[p] = BDD variable (= level) that holds bit p of a key (p = 0 is
             the least significant bit)
    The BDD is built bottom-up, one bit (= one BDD level) at a time, exactly like
    reducing a decision tree: a node whose two children are equal is skipped
    (elimination rule) and equal (level, low, high) triples are shared (merge rule).
    Only one ite call is made per node of the final ROBDD."""
    F = [mgr.false(), mgr.true()]  # node id -> BDD function; 0 = false, 1 = true
    P = np.unique(keys.astype(np.uint64))  # current distinct prefixes
    ids = np.ones(
        len(P), dtype=np.int64
    )  # sub-BDD below each prefix (all 'true' at start)
    for p in range(len(levels)):  # from the bottom bit up to the top bit
        bit = (P & np.uint64(1)).astype(bool)
        Q = P >> np.uint64(1)  # parent prefix
        uq, first = np.unique(Q, return_index=True)
        pos = np.searchsorted(uq, Q)
        low = np.zeros(len(uq), dtype=np.int64)  # missing child = false
        high = np.zeros(len(uq), dtype=np.int64)
        low[pos[~bit]] = ids[~bit]
        high[pos[bit]] = ids[bit]
        new_ids = low.copy()  # elimination: low == high -> skip the node
        need = low != high
        if need.any():
            pairs = np.stack([low[need], high[need]], axis=1)
            upairs, inv = np.unique(pairs, axis=0, return_inverse=True)  # merge rule
            base = len(F)
            x = mgr.var(levels[p])
            for lo, hi in upairs:
                F.append(x.ite(F[hi], F[lo]))
            new_ids[need] = base + inv.reshape(-1)
        P, ids = uq, new_ids
    return F[int(ids[0])] if len(ids) else mgr.false()


class Graph:
    def __init__(self, nodes_csv, edges_csv, threads=1):
        t0 = time.time()
        nodes = read_table(nodes_csv)
        self.ids = nodes.iloc[:, 0].to_numpy(dtype=np.int64)
        labels = nodes.iloc[:, -1].to_numpy(dtype=np.int64)
        edges = read_table(edges_csv, (0, 1)).to_numpy(dtype=np.int64)
        edges = np.unique(edges, axis=0)
        self.n_nodes, self.n_edges = len(self.ids), len(edges)
        self.edges = edges
        self.label_by_id = {
            int(node_id): int(label) for node_id, label in zip(self.ids, labels)
        }
        max_id = int(max(self.ids.max(), edges.max()))
        self.B = B = max(1, math.ceil(math.log2(max_id + 1)))

        self.mgr = mgr = BDDManager(100_000_000, 1_000_000, threads)
        mgr.add_vars(2 * B)
        # bit i of a node id (i = 0 least significant): x_i at level 2(B-1-i), y_i just below
        self.xvar = [2 * (B - 1 - i) for i in range(B)]
        self.yvar = [2 * (B - 1 - i) + 1 for i in range(B)]

        # T(x, y): interleave the bits of source and target into one key
        s, t = edges[:, 0].astype(np.uint64), edges[:, 1].astype(np.uint64)
        keys = np.zeros(len(edges), dtype=np.uint64)
        for i in range(B):
            keys |= ((s >> np.uint64(i)) & np.uint64(1)) << np.uint64(2 * i + 1)
            keys |= ((t >> np.uint64(i)) & np.uint64(1)) << np.uint64(2 * i)
        # key bit 2i holds y_i, key bit 2i+1 holds x_i
        key_levels = [v for i in range(B) for v in (self.yvar[i], self.xvar[i])]
        self.T = build_set(mgr, keys, key_levels)

        # L(x): the set of node ids with label 1
        self.L = build_set(mgr, self.ids[labels == 1], self.xvar)
        self.to_y = mgr.true().make_substitution(
            [(self.xvar[i], mgr.var(self.yvar[i])) for i in range(B)]
        )
        self.to_x = mgr.true().make_substitution(
            [(self.yvar[i], mgr.var(self.xvar[i])) for i in range(B)]
        )
        self.Ly = self.L.substitute(self.to_y)
        self.xcube = mgr.true()  # conjunction of all x variables
        for v in self.xvar:
            self.xcube = self.xcube & mgr.var(v)
        self.build_time = time.time() - t0

    def check_alternating_edges(self):
        """Inspect how many edges connect differently labelled nodes."""

        missing = set(np.unique(self.edges)) - set(self.label_by_id)

        if missing:
            raise ValueError(f"{len(missing)} edge endpoints have no node label")

        labels_source = np.array([self.label_by_id[int(s)] for s in self.edges[:, 0]])

        labels_target = np.array([self.label_by_id[int(t)] for t in self.edges[:, 1]])

        different = labels_source != labels_target

        print("Total edges:", len(self.edges))
        print("Edges with different labels:", np.count_nonzero(different))
        print("Edges with equal labels:", np.count_nonzero(~different))

    def build_alternating_relation(self):
        """Build a BDD from edges whose endpoint labels differ."""
        alt_edges = self.edges[
            np.array(
                [
                    self.label_by_id[int(s)] != self.label_by_id[int(t)]
                    for s, t in self.edges
                ]
            )
        ]

        s = alt_edges[:, 0].astype(np.uint64)
        t = alt_edges[:, 1].astype(np.uint64)

        keys = np.zeros(len(alt_edges), dtype=np.uint64)

        for i in range(self.B):
            keys |= ((s >> np.uint64(i)) & np.uint64(1)) << np.uint64(2 * i + 1)
            keys |= ((t >> np.uint64(i)) & np.uint64(1)) << np.uint64(2 * i)

        key_levels = [v for i in range(self.B) for v in (self.yvar[i], self.xvar[i])]

        return build_set(self.mgr, keys, key_levels)

    def node(self, s):
        """BDD over x of the single node s."""
        f = self.mgr.true()
        for i in range(self.B):
            v = self.mgr.var(self.xvar[i])
            f = f & (v if (s >> i) & 1 else ~v)
        return f

    def image(self, R, T):
        """Successors of the set R(x):  (exists x. R(x) and T(x, y)), renamed y -> x."""
        img = R.apply_exists(BooleanOperator.AND, T, self.xcube)
        return img.substitute(self.to_x)

    def reach(self, start, T):
        """All nodes reachable from the set 'start' (included), breadth-first on sets:
        repeatedly add the successors of the newly found nodes until nothing is new."""
        R, frontier, steps = start, start, 0
        while frontier.satisfiable():
            new = self.image(frontier, T) & ~R
            R, frontier, steps = R | new, new, steps + 1
        return R, steps

    def count(self, R):
        """Count satisfying assignments over the x variables only."""
        return R.sat_count(self.B)


def joint_node_count(*fs):
    """Number of distinct BDD nodes in several BDDs together (shared nodes once)."""
    seen, stack = set(), list(fs)
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        c = f.cofactors()
        if c is not None:
            stack.extend(c)
    return len(seen)


def main(nodes_csv, edges_csv, threads=1):
    G = Graph(nodes_csv, edges_csv, threads)
    G.check_alternating_edges()
    print(
        f"{nodes_csv}: {G.n_nodes} nodes, {G.n_edges} edges, {G.B} bits per node id, "
        f"BDDs built in {G.build_time:.1f} s"
    )
    print(f"  T  (transition relation)        : {G.T.node_count()} BDD nodes")
    print(f"  L  (labelling)                  : {G.L.node_count()} BDD nodes")
    print(f"  T and L together (shared once)  : {joint_node_count(G.T, G.L)} BDD nodes")

    # (b) plain reachability from node 1
    t0 = time.time()
    s0 = G.node(1)
    R, steps = G.reach(s0, G.T)
    print(
        f"(b) reachable from node 1         : {G.count(R)} nodes (node 1 included), "
        f"{steps} steps, {time.time() - t0:.1f} s"
    )
    print(f"    R  (reachable set)            : {R.node_count()} BDD nodes")

    # (c) s1 has label 1 and every following edge goes to a node with the other label
    t0 = time.time()
    T_alt = G.build_alternating_relation()  # label(x) != label(y)
    first = G.image(s0, G.T & G.Ly)  # successors of node 1 with label 1
    R_alt, steps = G.reach(first, T_alt)
    R_alt_all = R_alt | s0  # node 1 itself is s0
    print(
        f"(c) reachable alternatingly       : {G.count(R_alt_all)} nodes (node 1 included), "
        f"{steps} steps, {time.time() - t0:.1f} s"
    )
    print(f"    T_alt (label-changing edges)  : {T_alt.node_count()} BDD nodes")
    print(f"    R_alt (reachable set)         : {R_alt_all.node_count()} BDD nodes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
