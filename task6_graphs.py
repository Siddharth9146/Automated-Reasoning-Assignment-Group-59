import sys, time, math
import numpy as np
import pandas as pd
from oxidd.bdd import BDDManager
from oxidd.util import BooleanOperator


def read_table(path, cols=None):
    """Load a CSV or TSV file and optionally select specific columns."""
    df = pd.read_csv(path, sep=None, engine="python")

    if cols is not None:
        df = df.iloc[:, list(cols)]

    return df


def build_set(mgr, keys, levels):
    """Build a BDD representing a set of integer keys.

    Each key represents one element in the set. The list 'levels' maps
    each bit of a key to its corresponding BDD variable, starting with
    the least significant bit.

    The BDD is constructed from the lowest bit upwards. Nodes with
    identical children are removed, and identical nodes are shared.
    This keeps the resulting BDD reduced and avoids unnecessary nodes.
    """
    F = [mgr.false(), mgr.true()]
    P = np.unique(keys.astype(np.uint64))
    ids = np.ones(len(P), dtype=np.int64)

    for p in range(len(levels)):
        bit = (P & np.uint64(1)).astype(bool)
        Q = P >> np.uint64(1)

        uq, first = np.unique(Q, return_index=True)
        pos = np.searchsorted(uq, Q)

        # Missing children represent false.
        low = np.zeros(len(uq), dtype=np.int64)
        high = np.zeros(len(uq), dtype=np.int64)

        low[pos[~bit]] = ids[~bit]
        high[pos[bit]] = ids[bit]

        # Skip nodes whose children are identical.
        new_ids = low.copy()
        need = low != high

        if need.any():
            pairs = np.stack([low[need], high[need]], axis=1)
            upairs, inv = np.unique(pairs, axis=0, return_inverse=True)

            base = len(F)
            x = mgr.var(levels[p])

            # Reuse identical child pairs to avoid duplicate nodes.
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

        self.n_nodes = len(self.ids)
        self.n_edges = len(edges)
        self.edges = edges

        # Store each node's label for quick lookup.
        self.label_by_id = {
            int(node_id): int(label) for node_id, label in zip(self.ids, labels)
        }

        max_id = int(max(self.ids.max(), edges.max()))
        self.B = B = max(1, math.ceil(math.log2(max_id + 1)))

        # Create the BDD manager and allocate variables for both endpoints.
        self.mgr = mgr = BDDManager(100_000_000, 1_000_000, threads)
        mgr.add_vars(2 * B)

        # Interleave source and target variables at each bit position.
        self.xvar = [2 * (B - 1 - i) for i in range(B)]
        self.yvar = [2 * (B - 1 - i) + 1 for i in range(B)]

        # Encode each edge as a single integer key.
        s, t = edges[:, 0].astype(np.uint64), edges[:, 1].astype(np.uint64)
        keys = np.zeros(len(edges), dtype=np.uint64)

        for i in range(B):
            keys |= ((s >> np.uint64(i)) & np.uint64(1)) << np.uint64(2 * i + 1)
            keys |= ((t >> np.uint64(i)) & np.uint64(1)) << np.uint64(2 * i)

        # Even key bits store target bits; odd bits store source bits.
        key_levels = [v for i in range(B) for v in (self.yvar[i], self.xvar[i])]
        self.T = build_set(mgr, keys, key_levels)

        # Build the BDD for nodes with label 1.
        self.L = build_set(mgr, self.ids[labels == 1], self.xvar)

        # Substitutions used to switch between source and target variables.
        self.to_y = mgr.true().make_substitution(
            [(self.xvar[i], mgr.var(self.yvar[i])) for i in range(B)]
        )
        self.to_x = mgr.true().make_substitution(
            [(self.yvar[i], mgr.var(self.xvar[i])) for i in range(B)]
        )

        self.Ly = self.L.substitute(self.to_y)

        # Used when eliminating all source variables during image computation.
        self.xcube = mgr.true()
        for v in self.xvar:
            self.xcube = self.xcube & mgr.var(v)

        self.build_time = time.time() - t0

    def check_alternating_edges(self):
        """Count edges whose endpoints have different labels."""
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
        """Build a BDD containing only edges that change the node label."""
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
        """Return a BDD representing a single node ID."""
        f = self.mgr.true()

        for i in range(self.B):
            v = self.mgr.var(self.xvar[i])
            f = f & (v if (s >> i) & 1 else ~v)

        return f

    def image(self, R, T):
        """Find all successors of R using the transition relation T."""
        img = R.apply_exists(BooleanOperator.AND, T, self.xcube)
        return img.substitute(self.to_x)

    def reach(self, start, T):
        """Find every node reachable from the starting set.

        Expand the current frontier one step at a time until no new
        nodes can be reached.
        """
        R, frontier, steps = start, start, 0

        while frontier.satisfiable():
            new = self.image(frontier, T) & ~R
            R, frontier, steps = R | new, new, steps + 1

        return R, steps

    def count(self, R):
        """Count satisfying assignments over the node variables."""
        return R.sat_count(self.B)


def joint_node_count(*fs):
    """Count distinct BDD nodes across several BDDs, counting shared nodes once."""
    seen = set()
    stack = list(fs)

    while stack:
        f = stack.pop()

        if f in seen:
            continue

        seen.add(f)
        children = f.cofactors()

        if children is not None:
            stack.extend(children)

    return len(seen)


def main(nodes_csv, edges_csv, threads=1):
    G = Graph(nodes_csv, edges_csv, threads)
    G.check_alternating_edges()

    print(
        f"{nodes_csv}: {G.n_nodes} nodes, {G.n_edges} edges"
        f"{G.B} bits per node ID, "
        f"BDDs built in {G.build_time:.1f} s"
    )
    print(f"  T  (transition relation)        : {G.T.node_count()} BDD nodes")
    print(f"  L  (labelling)                  : {G.L.node_count()} BDD nodes")
    print(f"  T and L together (shared once)  : {joint_node_count(G.T, G.L)} BDD nodes")

    # (b) Find all nodes reachable from node 1.
    t0 = time.time()
    s0 = G.node(1)
    R, steps = G.reach(s0, G.T)

    print(
        f"(b) reachable from node 1         : {G.count(R)} nodes (node 1 included), "
        f"{steps} steps, {time.time() - t0:.1f} s"
    )
    print(f"    R  (reachable set)            : {R.node_count()} BDD nodes")

    # (c) Start at a label-1 successor of node 1, then follow only
    # edges that connect nodes with different labels.
    t0 = time.time()
    T_alt = G.build_alternating_relation()
    first = G.image(s0, G.T & G.Ly)

    R_alt, steps = G.reach(first, T_alt)
    R_alt_all = R_alt | s0

    print(
        f"(c) reachable alternatingly       : {G.count(R_alt_all)} nodes (node 1 included), "
        f"{steps} steps, {time.time() - t0:.1f} s"
    )
    print(f"    T_alt (label-changing edges)  : {T_alt.node_count()} BDD nodes")
    print(f"    R_alt (reachable set)         : {R_alt_all.node_count()} BDD nodes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
