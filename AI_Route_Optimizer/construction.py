"""
construction.py – hand-written construction heuristics (v2.0)
=============================================================

Every method is written out step by step, following the classic papers (see
MANUAL_METHODS.md). No solver library is used. When a Tracer is given, each
method writes its intermediate steps to a trace file ("show your work").

  clarke_wright_parallel   Clarke & Wright (1964) savings, parallel version,
                           with the Paessens (1988) parameters lambda / mu
  clarke_wright_sequential Webb (1964) sequential savings
  sweep                    Gillett & Miller (1974) sweep
  nearest_neighbour        greedy nearest-neighbour fill (as in vss2sn/cvrp)
  cheapest_insertion       sequential cheapest insertion (van Breedam 1994)

What these methods lack in most reference repos, and what we added:
  * The route count may exceed the fleet size K. We report it here, and the
    solver then repairs it (Split with K routes + penalised local search).
  * The sequential savings method may leave customers unserved. We restart a
    new route until every customer is served.
"""
from __future__ import annotations

import math


class Tracer:
    """Collects the step-by-step log of a run (written to trace.txt)."""

    def __init__(self, enabled: bool = True, max_lines_per_section: int = 60):
        self.enabled = enabled
        self.lines: list[str] = []
        self.max_lines = max_lines_per_section
        self._count = 0

    def section(self, title: str):
        if self.enabled:
            self.lines.append("")
            self.lines.append("=" * 72)
            self.lines.append(title)
            self.lines.append("=" * 72)
            self._count = 0

    def log(self, text: str, force: bool = False):
        if not self.enabled:
            return
        if force or self._count < self.max_lines:
            self.lines.append(text)
        elif self._count == self.max_lines:
            self.lines.append("   ... (further steps of this section omitted)")
        self._count += 1

    def save(self, path):
        if self.enabled:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.lines) + "\n")


def _cost(routes, D):
    s = 0.0
    for r in routes:
        if r:
            s += D[0][r[0]] + D[r[-1]][0] + sum(D[a][b] for a, b in zip(r, r[1:]))
    return s


# --------------------------------------------------------------------------
# 1. Clarke & Wright parallel savings  (+ Paessens parameters)
# --------------------------------------------------------------------------
def clarke_wright_parallel(inst, lam: float = 1.0, mu: float = 0.0, tracer: Tracer | None = None):
    """
    Step 1  start with one route per customer:  0 -> i -> 0
    Step 2  savings  s(i,j) = d(0,i) + d(0,j) - lam*d(i,j) + mu*|d(0,i) - d(0,j)|
    Step 3  sort the savings from largest to smallest
    Step 4  for each pair (i, j) in that order, merge the route ending in i
            with the route starting in j when
              - i and j are in different routes,
              - i and j are both route END points (not interior),
              - the merged load <= Q
    Step 5  stop when the list is exhausted
    """
    D, q, Q, n = inst.dist, inst.demand, inst.capacity, inst.n_customers
    tr = tracer or Tracer(False)
    tr.section(f"CLARKE-WRIGHT PARALLEL SAVINGS  (lambda={lam}, mu={mu})")

    route_of = {i: i for i in range(1, n + 1)}
    routes = {i: [i] for i in range(1, n + 1)}
    load = {i: q[i] for i in range(1, n + 1)}
    tr.log(f"Step 1: {n} initial routes 0-i-0, total distance {_cost(routes.values(), D):.0f}")

    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = D[0][i] + D[0][j] - lam * D[i][j] + mu * abs(D[0][i] - D[0][j])
            if s > 0:
                savings.append((s, i, j))
    savings.sort(key=lambda t: -t[0])
    tr.log(f"Step 2-3: {len(savings)} positive savings, sorted. Top 10:")
    for s, i, j in savings[:10]:
        tr.log(f"   s({i},{j}) = {D[0][i]:.0f} + {D[0][j]:.0f} - {lam}*{D[i][j]:.0f}"
               f"{' + ' + str(mu) + '*|..|' if mu else ''} = {s:.1f}")

    tr.log("Step 4: merges")
    step = 0
    for s, i, j in savings:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        if load[ri] + load[rj] > Q:
            continue
        A, B = routes[ri], routes[rj]
        # i and j must be end points; orient so that A ends with i and B starts with j
        if A[-1] == i:
            pass
        elif A[0] == i:
            A = A[::-1]
        else:
            continue
        if B[0] == j:
            pass
        elif B[-1] == j:
            B = B[::-1]
        else:
            continue
        merged = A + B
        routes[ri] = merged
        load[ri] += load[rj]
        for c in B:
            route_of[c] = ri
        del routes[rj], load[rj]
        step += 1
        tr.log(f"   merge #{step}: link {i}-{j} (saving {s:.1f}) -> route load {load[ri]:.0f}/{Q:.0f}, "
               f"{len(routes)} routes left")

    result = [r for r in routes.values()]
    tr.log(f"Result: {len(result)} routes, distance {_cost(result, D):.0f}"
           f"{'  (more routes than the fleet K=' + str(inst.num_vehicles) + ': repaired later)' if len(result) > inst.num_vehicles else ''}",
           force=True)
    return result


# --------------------------------------------------------------------------
# 2. Webb (1964) sequential savings
# --------------------------------------------------------------------------
def clarke_wright_sequential(inst, tracer: Tracer | None = None):
    """
    Build ONE route at a time: seed it with the best saving pair of unserved
    customers, then keep extending it at either end with the best saving
    until nothing fits, then start the next route. Loop until everyone is
    served (the original method may leave customers out; we don't).
    """
    D, q, Q, n = inst.dist, inst.demand, inst.capacity, inst.n_customers
    tr = tracer or Tracer(False)
    tr.section("CLARKE-WRIGHT SEQUENTIAL SAVINGS")
    unserved = set(range(1, n + 1))
    routes = []
    while unserved:
        # seed: best saving pair (or farthest single customer)
        best, seed = -1, None
        for i in unserved:
            for j in unserved:
                if i < j and q[i] + q[j] <= Q:
                    s = D[0][i] + D[0][j] - D[i][j]
                    if s > best:
                        best, seed = s, [i, j]
        if seed is None:
            seed = [max(unserved, key=lambda c: D[0][c])]
        route = seed[:]
        for c in seed:
            unserved.discard(c)
        ld = sum(q[c] for c in route)
        while True:
            cand = None
            for c in unserved:
                if ld + q[c] > Q:
                    continue
                s_front = D[0][c] + D[0][route[0]] - D[c][route[0]]
                s_back = D[0][route[-1]] + D[0][c] - D[route[-1]][c]
                s, end = (s_front, 0) if s_front >= s_back else (s_back, 1)
                if cand is None or s > cand[0]:
                    cand = (s, c, end)
            if cand is None or cand[0] <= 0:
                break
            _, c, end = cand
            route = [c] + route if end == 0 else route + [c]
            ld += q[c]
            unserved.discard(c)
        routes.append(route)
        tr.log(f"   route {len(routes)}: {route}  load {ld:.0f}/{Q:.0f}")
    tr.log(f"Result: {len(routes)} routes, distance {_cost(routes, D):.0f}", force=True)
    return routes


# --------------------------------------------------------------------------
# 3. Sweep
# --------------------------------------------------------------------------
def sweep(inst, start_angle: float = 0.0, clockwise: bool = False, tracer: Tracer | None = None):
    """Sort customers by polar angle around the depot; fill trucks in that order."""
    D, q, Q = inst.dist, inst.demand, inst.capacity
    x0, y0 = inst.coords[0]
    tr = tracer or Tracer(False)
    tr.section(f"SWEEP (start angle {math.degrees(start_angle):.0f} deg, "
               f"{'clockwise' if clockwise else 'counter-clockwise'})")

    def ang(c):
        a = math.atan2(inst.coords[c][1] - y0, inst.coords[c][0] - x0) - start_angle
        a = a % (2 * math.pi)
        return -a if clockwise else a

    order = sorted(range(1, inst.n_customers + 1), key=ang)
    routes, cur, ld = [], [], 0.0
    for c in order:
        if ld + q[c] > Q:
            routes.append(cur)
            tr.log(f"   route {len(routes)} closed at load {ld:.0f}: {cur}")
            cur, ld = [], 0.0
        cur.append(c)
        ld += q[c]
    if cur:
        routes.append(cur)
        tr.log(f"   route {len(routes)} closed at load {ld:.0f}: {cur}")
    tr.log(f"Result: {len(routes)} routes, distance {_cost(routes, D):.0f}", force=True)
    return routes


# --------------------------------------------------------------------------
# 4. Nearest neighbour
# --------------------------------------------------------------------------
def nearest_neighbour(inst, tracer: Tracer | None = None):
    D, q, Q = inst.dist, inst.demand, inst.capacity
    tr = tracer or Tracer(False)
    tr.section("NEAREST NEIGHBOUR")
    left, routes = set(range(1, inst.n_customers + 1)), []
    while left:
        cur, ld, pos = [], 0.0, 0
        while True:
            cand = [c for c in left if ld + q[c] <= Q]
            if not cand:
                break
            nxt = min(cand, key=lambda c: (D[pos][c], c))
            cur.append(nxt)
            ld += q[nxt]
            left.remove(nxt)
            pos = nxt
        routes.append(cur)
        tr.log(f"   route {len(routes)}: {cur}  load {ld:.0f}")
    tr.log(f"Result: {len(routes)} routes, distance {_cost(routes, D):.0f}", force=True)
    return routes


# --------------------------------------------------------------------------
# 5. Sequential cheapest insertion
# --------------------------------------------------------------------------
def cheapest_insertion(inst, tracer: Tracer | None = None):
    """Seed each route with the farthest unserved customer, then repeatedly
    insert the customer with the cheapest feasible insertion cost."""
    D, q, Q = inst.dist, inst.demand, inst.capacity
    tr = tracer or Tracer(False)
    tr.section("SEQUENTIAL CHEAPEST INSERTION")
    left, routes = set(range(1, inst.n_customers + 1)), []
    while left:
        seed = max(left, key=lambda c: (D[0][c], c))
        route, ld = [seed], q[seed]
        left.remove(seed)
        while True:
            best = None
            for c in left:
                if ld + q[c] > Q:
                    continue
                seq = [0] + route + [0]
                for p in range(len(seq) - 1):
                    a, b = seq[p], seq[p + 1]
                    inc = D[a][c] + D[c][b] - D[a][b]
                    if best is None or inc < best[0]:
                        best = (inc, c, p)
            if best is None:
                break
            _, c, p = best
            route.insert(p, c)
            ld += q[c]
            left.remove(c)
        routes.append(route)
        tr.log(f"   route {len(routes)}: {route}  load {ld:.0f}")
    tr.log(f"Result: {len(routes)} routes, distance {_cost(routes, D):.0f}", force=True)
    return routes


def construction_portfolio(inst, tracer: Tracer | None = None):
    """All hand-written constructions used to seed the population (name, routes)."""
    out = [("CW parallel", clarke_wright_parallel(inst, 1.0, 0.0, tracer))]
    # Paessens parameter grid (quiet: only the standard version is traced in full)
    for lam in (0.6, 0.8, 1.2, 1.4):
        for mu in (0.0, 0.5):
            out.append((f"CW lambda={lam} mu={mu}", clarke_wright_parallel(inst, lam, mu)))
    out.append(("CW sequential", clarke_wright_sequential(inst, tracer)))
    out.append(("Sweep", sweep(inst, 0.0, False, tracer)))
    for k in range(1, 6):
        out.append((f"Sweep {k * 60} deg", sweep(inst, math.radians(k * 60), k % 2 == 0)))
    out.append(("Nearest neighbour", nearest_neighbour(inst, tracer)))
    out.append(("Cheapest insertion", cheapest_insertion(inst, tracer)))
    return out
