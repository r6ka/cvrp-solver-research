"""
local_search2.py – v2.0 local search with hand-derived O(1) move evaluation
===========================================================================

v1.0 rebuilt each candidate route and recomputed its whole cost (O(route
length) per move). This was the main speed bottleneck in the 29 Sep stress
test. Here every move's cost change is written out by hand from the edges it
removes and adds, so evaluating a move costs O(1). A route is only rebuilt
when a move is actually applied.

Notation for every move (depot = 0):
    u in route A at position i,  pu = pred(u),  x = succ(u)
    v in route B at position j,  pv = pred(v),  y = succ(v)

  RELOCATE  u after v   delta = d(pu,x) - d(pu,u) - d(u,x) + d(v,u) + d(u,y) - d(v,y)
  RELOCATE  u before v  delta = d(pu,x) - d(pu,u) - d(u,x) + d(pv,u) + d(u,v) - d(pv,v)
  OR-OPT    (u,x) after v     (both orientations)
  SWAP      u <-> v     delta = d(pu,v)+d(v,x)-d(pu,u)-d(u,x) + d(pv,u)+d(u,y)-d(pv,v)-d(v,y)
  2-OPT     (same route) reverse x..v   delta = d(u,v) + d(x,y) - d(u,x) - d(v,y)
  2-OPT*    tails swap  A[:i]+B[j+1:], B[:j]+A[i+1:]      delta = d(u,y) + d(v,x) - d(u,x) - d(v,y)
  2-OPT*    reversed    A[:i]+rev(B[:j]), rev(A[i+1:])+B[j+1:]  delta = d(u,v) + d(x,y) - d(u,x) - d(v,y)
  SWAP*     (Vidal 2022) exchange u and v between two routes, each re-inserted
            at its BEST position in the other route (not in place), using the
            3 best insertion positions pre-computed per customer and route.

The capacity part of the penalised cost is  P * max(0, load - Q)  per route,
using cached route loads and prefix loads (for 2-opt*).

Every O(1) formula is checked against a full recomputation in
tests/test_moves.py (random instances, all move types).
"""
from __future__ import annotations

import random


class FastLocalSearch:
    def __init__(self, inst, granularity: int = 20, rng: random.Random | None = None):
        self.inst = inst
        self.D = inst.dist
        self.q = inst.demand
        self.Q = inst.capacity
        self.K = inst.num_vehicles
        self.n = inst.n_customers
        self.rng = rng or random.Random(0)
        D = self.D
        n = self.n
        self.neigh = []
        for u in range(n + 1):
            if u == 0:
                self.neigh.append([])
                continue
            others = sorted((D[u][v], v) for v in range(1, n + 1) if v != u)
            self.neigh.append([v for _, v in others[:granularity]])
        self.move_counts = {}
        self.debug_check = False

    # ------------------------------------------------------------ route cache
    def _refresh(self, r):
        R = self.routes[r]
        q, D = self.q, self.D
        pre, s = [], 0.0
        for c in R:
            s += q[c]
            pre.append(s)
        self.prefix[r] = pre
        self.load[r] = s
        if R:
            c = D[0][R[0]] + D[R[-1]][0]
            for a, b in zip(R, R[1:]):
                c += D[a][b]
        else:
            c = 0.0
        self.dist[r] = c
        for i, cst in enumerate(R):
            self.where[cst] = (r, i)

    def _pen(self, load):
        e = load - self.Q
        return self.penalty * e if e > 1e-9 else 0.0

    def _pred(self, R, i):
        return R[i - 1] if i > 0 else 0

    def _succ(self, R, i):
        return R[i + 1] if i + 1 < len(R) else 0

    def _apply(self, name, changes):
        if self.debug_check:
            before = self.total()
            old_set = sorted(c for r in changes for c in self.routes[r])
        for r, nr in changes.items():
            self.routes[r] = nr
        self.nmoves += 1
        for r in changes:
            self._refresh(r)
            self.mod[r] = self.nmoves
        self.move_counts[name] = self.move_counts.get(name, 0) + 1
        if self.debug_check:
            after = self.total()   # recomputed from scratch by _refresh
            new_set = sorted(c for r in changes for c in self.routes[r])
            assert new_set == old_set, f"{name}: customers lost or duplicated"
            assert after < before - 1e-9, f"{name}: predicted improvement but {before} -> {after}"
        return True

    def total(self):
        return sum(self.dist) + sum(self._pen(l) for l in self.load)

    # ------------------------------------------------------------ node moves
    def _node_moves(self, u, loop) -> bool:
        D, q = self.D, self.q
        routes, where, load, prefix, mod = self.routes, self.where, self.load, self.prefix, self.mod
        P, Q = self.penalty, self.Q
        last = self.last_test[u]
        self.last_test[u] = self.nmoves
        Du = D[u]
        qu = q[u]
        for v in self.neigh[u]:
            ra, i = where[u]
            rb, j = where[v]
            # skip: nothing changed in either route since u was last tested
            if loop > 0 and mod[ra] <= last and mod[rb] <= last:
                continue
            A, B = routes[ra], routes[rb]
            pu = A[i - 1] if i > 0 else 0
            x = A[i + 1] if i + 1 < len(A) else 0
            pv = B[j - 1] if j > 0 else 0
            y = B[j + 1] if j + 1 < len(B) else 0
            Dv = D[v]
            rem_u = D[pu][x] - D[pu][u] - Du[x]

            if ra != rb:
                la, lb = load[ra], load[rb]
                qv = q[v]
                pen_old = (P * (la - Q) if la > Q else 0.0) + (P * (lb - Q) if lb > Q else 0.0)
                # RELOCATE u after v / before v
                t1, t2 = la - qu, lb + qu
                dpen = (P * (t1 - Q) if t1 > Q else 0.0) + (P * (t2 - Q) if t2 > Q else 0.0) - pen_old
                d1 = rem_u + Dv[u] + Du[y] - Dv[y]
                if d1 + dpen < -1e-9:
                    return self._apply("relocate", {ra: A[:i] + A[i + 1:], rb: B[:j + 1] + [u] + B[j + 1:]})
                d2 = rem_u + D[pv][u] + Du[v] - D[pv][v]
                if d2 + dpen < -1e-9:
                    return self._apply("relocate", {ra: A[:i] + A[i + 1:], rb: B[:j] + [u] + B[j:]})
                # SWAP u <-> v
                t1, t2 = la - qu + qv, lb - qv + qu
                dpen = (P * (t1 - Q) if t1 > Q else 0.0) + (P * (t2 - Q) if t2 > Q else 0.0) - pen_old
                d = (D[pu][v] + Dv[x] - D[pu][u] - Du[x]
                     + D[pv][u] + Du[y] - D[pv][v] - Dv[y])
                if d + dpen < -1e-9:
                    A2 = A[:]; B2 = B[:]
                    A2[i], B2[j] = v, u
                    return self._apply("swap", {ra: A2, rb: B2})
                # OR-OPT: move (u, x) after v
                if x != 0:
                    xx = A[i + 2] if i + 2 < len(A) else 0
                    rem_s = D[pu][xx] - D[pu][u] - D[x][xx]
                    t1, t2 = la - qu - q[x], lb + qu + q[x]
                    dpen = (P * (t1 - Q) if t1 > Q else 0.0) + (P * (t2 - Q) if t2 > Q else 0.0) - pen_old
                    d = rem_s + Dv[u] + D[x][y] - Dv[y]
                    if d + dpen < -1e-9:
                        return self._apply("or-opt", {ra: A[:i] + A[i + 2:], rb: B[:j + 1] + [u, x] + B[j + 1:]})
                    d = rem_s + Dv[x] + Du[y] - Dv[y]
                    if d + dpen < -1e-9:
                        return self._apply("or-opt", {ra: A[:i] + A[i + 2:], rb: B[:j + 1] + [x, u] + B[j + 1:]})
                # 2-OPT* (tails exchange): u -> y, v -> x
                pa, pb = prefix[ra][i], prefix[rb][j]
                t1, t2 = pa + (lb - pb), pb + (la - pa)
                d = Du[y] + Dv[x] - Du[x] - Dv[y]
                if d + (P * (t1 - Q) if t1 > Q else 0.0) + (P * (t2 - Q) if t2 > Q else 0.0) - pen_old < -1e-9:
                    return self._apply("2-opt*", {ra: A[:i + 1] + B[j + 1:], rb: B[:j + 1] + A[i + 1:]})
                # 2-OPT* (reversed pieces): u -> v and x -> y
                t1, t2 = pa + pb, (la - pa) + (lb - pb)
                d = Du[v] + D[x][y] - Du[x] - Dv[y]
                if d + (P * (t1 - Q) if t1 > Q else 0.0) + (P * (t2 - Q) if t2 > Q else 0.0) - pen_old < -1e-9:
                    return self._apply("2-opt*", {ra: A[:i + 1] + B[:j + 1][::-1],
                                                  rb: A[i + 1:][::-1] + B[j + 1:]})
            else:
                # same route: distance-only moves (load unchanged)
                if v != pu:
                    d = rem_u + Dv[u] + Du[y] - Dv[y]
                    if d < -1e-9:
                        R = A[:i] + A[i + 1:]
                        k = R.index(v)
                        return self._apply("relocate", {ra: R[:k + 1] + [u] + R[k + 1:]})
                if v != x:
                    d = rem_u + D[pv][u] + Du[v] - D[pv][v]
                    if d < -1e-9:
                        R = A[:i] + A[i + 1:]
                        k = R.index(v)
                        return self._apply("relocate", {ra: R[:k] + [u] + R[k:]})
                # 2-OPT: make u -> v an edge by reversing the segment in between
                a, b = (i, j) if i < j else (j, i)
                if b > a + 1:
                    ua, ub = A[a], A[b]
                    xa = A[a + 1]
                    yb = A[b + 1] if b + 1 < len(A) else 0
                    d = D[ua][ub] + D[xa][yb] - D[ua][xa] - D[ub][yb]
                    if d < -1e-9:
                        return self._apply("2-opt", {ra: A[:a + 1] + A[a + 1:b + 1][::-1] + A[b + 1:]})
                    pa_ = A[a - 1] if a > 0 else 0
                    d = D[pa_][ub] + D[ua][yb] - D[pa_][ua] - D[ub][yb]
                    if d < -1e-9:
                        return self._apply("2-opt", {ra: A[:a] + A[a:b + 1][::-1] + A[b + 1:]})
                # SWAP inside the route (non-adjacent)
                if abs(i - j) > 1:
                    d = (D[pu][v] + Dv[x] - D[pu][u] - Du[x]
                         + D[pv][u] + Du[y] - D[pv][v] - Dv[y])
                    if d < -1e-9:
                        R = A[:]
                        R[i], R[j] = v, u
                        return self._apply("swap", {ra: R})

        # move u alone into an empty (unused) vehicle
        ra, i = self.where[u]
        A = routes[ra]
        if len(A) > 1:
            for re, R in enumerate(routes):
                if not R:
                    pu, x = self._pred(A, i), self._succ(A, i)
                    d = D[pu][x] - D[pu][u] - D[u][x] + D[0][u] + D[u][0]
                    dpen = self._pen(self.load[ra] - q[u]) - self._pen(self.load[ra])
                    if d + dpen < -1e-9:
                        return self._apply("new-route", {ra: A[:i] + A[i + 1:], re: [u]})
                    break
        return False

    # ----------------------------------------------------------------- SWAP*
    def _top3_insertions(self, u, R):
        """3 cheapest positions to insert u into route R: list of (cost, a, b)."""
        D = self.D
        best = []
        prev = 0
        for c in R + [0]:
            cost = D[prev][u] + D[u][c] - D[prev][c]
            best.append((cost, prev, c))
            prev = c
        best.sort(key=lambda t: t[0])
        return best[:3]

    def _swap_star_pair(self, ra, rb) -> bool:
        D, q = self.D, self.q
        A, B = self.routes[ra], self.routes[rb]
        if not A or not B:
            return False
        la, lb = self.load[ra], self.load[rb]
        pen_old = self._pen(la) + self._pen(lb)
        insA = {u: self._top3_insertions(u, B) for u in A}   # u from A into B
        insB = {v: self._top3_insertions(v, A) for v in B}
        best = None
        for i, u in enumerate(A):
            pu, xu = self._pred(A, i), self._succ(A, i)
            rem_u = D[pu][xu] - D[pu][u] - D[u][xu]
            for j, v in enumerate(B):
                dpen = self._pen(la - q[u] + q[v]) + self._pen(lb - q[v] + q[u]) - pen_old
                pv, yv = self._pred(B, j), self._succ(B, j)
                rem_v = D[pv][yv] - D[pv][v] - D[v][yv]
                # best insertion of u into B \ {v}
                cu = D[pv][u] + D[u][yv] - D[pv][yv]
                eu = (pv, yv)
                for c, a, b in insA[u]:
                    if a != v and b != v:
                        if c < cu:
                            cu, eu = c, (a, b)
                        break
                # best insertion of v into A \ {u}
                cv = D[pu][v] + D[v][xu] - D[pu][xu]
                ev = (pu, xu)
                for c, a, b in insB[v]:
                    if a != u and b != u:
                        if c < cv:
                            cv, ev = c, (a, b)
                        break
                delta = rem_u + rem_v + cu + cv + dpen
                if delta < -1e-9 and (best is None or delta < best[0]):
                    best = (delta, u, v, eu, ev)
        if best is None:
            return False
        _, u, v, eu, ev = best
        A2 = [c for c in A if c != u]
        B2 = [c for c in B if c != v]
        B2 = self._insert_between(B2, u, *eu)
        A2 = self._insert_between(A2, v, *ev)
        return self._apply("SWAP*", {ra: A2, rb: B2})

    @staticmethod
    def _insert_between(R, c, a, b):
        if a == 0:                      # edge (depot, first customer)
            return [c] + R
        k = R.index(a)
        return R[:k + 1] + [c] + R[k + 1:]

    def _swap_star(self) -> bool:
        # candidate route pairs: routes containing granular neighbours
        pairs = set()
        for u in range(1, self.n + 1):
            ru = self.where[u][0]
            for v in self.neigh[u][:6]:
                rv = self.where[v][0]
                if ru != rv:
                    pairs.add((min(ru, rv), max(ru, rv)))
        improved = False
        for ra, rb in pairs:
            key = (ra, rb)
            if self.ss_last.get(key, -1) >= max(self.mod[ra], self.mod[rb]):
                continue          # both routes unchanged since this pair was tested
            self.ss_last[key] = self.nmoves
            if self._swap_star_pair(ra, rb):
                improved = True
        return improved

    # ------------------------------------------------------------------ main
    def run(self, routes, penalty: float = 1e9, use_swap_star: bool = True):
        self.penalty = penalty
        self.routes = [list(r) for r in routes if r]
        while len(self.routes) < self.K:
            self.routes.append([])
        m = len(self.routes)
        self.prefix = [None] * m
        self.load = [0.0] * m
        self.dist = [0.0] * m
        self.where = {}
        self.nmoves = 0
        self.mod = [0] * m
        self.last_test = [-1] * (self.n + 1)
        self.ss_last = {}
        for r in range(m):
            self._refresh(r)

        customers = list(range(1, self.n + 1))
        loop = 0
        while True:
            improved = False
            self.rng.shuffle(customers)
            for u in customers:
                if self._node_moves(u, loop):
                    improved = True
            loop += 1
            if not improved and use_swap_star:
                improved = self._swap_star()
            if not improved:
                break
        return [r for r in self.routes if r]
