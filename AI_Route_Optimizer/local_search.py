"""
local_search.py
---------------
Local-search "education" step of the memetic (hybrid) Genetic Algorithm.

It repeatedly applies the first move that lowers the penalised objective
    cost = objective(route) + penalty * max(0, load - capacity)
(capacity excess is penalised, so the GA can repair infeasible children):

  * Relocate      – move one customer to another position / route
  * Or-opt(2)     – move two consecutive customers
  * Swap          – exchange two customers
  * 2-opt         – reverse a segment inside one route
  * 2-opt*        – exchange the tails of two different routes
  * New route     – move a customer to an unused vehicle (if the fleet allows)
  * Reversal      – drive a route backwards (only matters for the CO2 objective)

To stay fast, each customer u is only combined with its `granularity` nearest
neighbours v ("granular neighbourhood"), a standard technique in modern
vehicle-routing heuristics.
"""

from __future__ import annotations

import random


def nearest_neighbours(D, n_customers: int, k: int):
    neigh = [[] for _ in range(n_customers + 1)]
    for u in range(1, n_customers + 1):
        others = sorted((D[u][v], v) for v in range(1, n_customers + 1) if v != u)
        neigh[u] = [v for _, v in others[:k]]
    return neigh


class LocalSearch:
    def __init__(self, inst, objective, granularity: int = 12, rng: random.Random | None = None):
        self.inst = inst
        self.obj = objective
        self.q = inst.demand
        self.Q = inst.capacity
        self.K = inst.num_vehicles
        self.rng = rng or random.Random(0)
        self.neigh = nearest_neighbours(inst.dist, inst.n_customers, granularity)

    # ------------------------------------------------------------------ helpers
    def _load(self, r):
        q = self.q
        return sum(q[c] for c in r)

    def _index(self):
        self.where = {}
        for ri, r in enumerate(self.routes):
            for pi, c in enumerate(r):
                self.where[c] = (ri, pi)

    def _pcost(self, r) -> float:
        """Route cost + penalty for exceeding the vehicle capacity."""
        excess = self._load(r) - self.Q
        c = self.obj.route_cost(r)
        return c + self.penalty * excess if excess > 1e-9 else c

    def _try(self, changes: dict) -> bool:
        """changes = {route_index: new_route}. Apply if the penalised cost improves."""
        old = 0.0
        new = 0.0
        new_costs = {}
        for ri, nr in changes.items():
            c = self._pcost(nr)
            new_costs[ri] = c
            new += c
            old += self.costs[ri]
        if new < old - 1e-9:
            for ri, nr in changes.items():
                self.routes[ri] = nr
                self.costs[ri] = new_costs[ri]
            self._index()
            return True
        return False

    # -------------------------------------------------------------------- moves
    def _moves_for(self, u) -> bool:
        routes = self.routes
        ru, iu = self.where[u]
        A = routes[ru]

        for v in self.neigh[u]:
            ru, iu = self.where[u]
            A = routes[ru]
            rv, jv = self.where[v]
            B = routes[rv]

            # ---- Relocate u after v and before v
            if ru != rv:
                A2 = A[:iu] + A[iu + 1:]
                if self._try({ru: A2, rv: B[:jv + 1] + [u] + B[jv + 1:]}):
                    return True
                if self._try({ru: A2, rv: B[:jv] + [u] + B[jv:]}):
                    return True
            else:
                R = A[:iu] + A[iu + 1:]
                jj = R.index(v)
                for pos in (jj + 1, jj):
                    R2 = R[:pos] + [u] + R[pos:]
                    if R2 != A and self._try({ru: R2}):
                        return True

            # ---- Or-opt: move (u, next(u)) after v
            if iu + 1 < len(A):
                x = A[iu + 1]
                if v != x:
                    if ru != rv:
                        A2 = A[:iu] + A[iu + 2:]
                        for seg in ([u, x], [x, u]):
                            if self._try({ru: A2, rv: B[:jv + 1] + seg + B[jv + 1:]}):
                                return True
                    else:
                        R = A[:iu] + A[iu + 2:]
                        jj = R.index(v)
                        for seg in ([u, x], [x, u]):
                            R2 = R[:jj + 1] + seg + R[jj + 1:]
                            if R2 != A and self._try({ru: R2}):
                                return True

            # ---- Swap u and v
            if ru != rv:
                A2 = A[:iu] + [v] + A[iu + 1:]
                B2 = B[:jv] + [u] + B[jv + 1:]
                if self._try({ru: A2, rv: B2}):
                    return True
            else:
                R = A[:]
                R[iu], R[jv] = R[jv], R[iu]
                if self._try({ru: R}):
                    return True

            # ---- 2-opt (same route) / 2-opt* (two routes)
            if ru == rv:
                i, j = (iu, jv) if iu < jv else (jv, iu)
                R = A[:i + 1] + A[i + 1:j + 1][::-1] + A[j + 1:]
                if R != A and self._try({ru: R}):
                    return True
                R = A[:i] + A[i:j + 1][::-1] + A[j + 1:]
                if R != A and self._try({ru: R}):
                    return True
            else:
                # connect u -> next(v) ... and v -> next(u) (tail exchange)
                A2 = A[:iu + 1] + B[jv + 1:]
                B2 = B[:jv + 1] + A[iu + 1:]
                if self._try({ru: A2, rv: B2}):
                    return True
                # connect u -> v (reverse one piece of each route)
                A2 = A[:iu + 1] + B[:jv + 1][::-1]
                B2 = A[iu + 1:][::-1] + B[jv + 1:]
                if self._try({ru: A2, rv: B2}):
                    return True

        # ---- Move u alone to an unused vehicle
        ru, iu = self.where[u]
        A = routes[ru]
        if len(A) > 1:
            empty = [ri for ri, r in enumerate(routes) if not r]
            if empty:
                if self._try({ru: A[:iu] + A[iu + 1:], empty[0]: [u]}):
                    return True
        return False

    def _reversals(self) -> bool:
        if self.obj.symmetric:
            return False
        improved = False
        for ri, r in enumerate(self.routes):
            if len(r) > 1 and self._try({ri: r[::-1]}):
                improved = True
        return improved

    # --------------------------------------------------------------------- main
    def run(self, routes, penalty: float = 1e9):
        """
        penalty = cost per unit of capacity excess.  A huge penalty means
        "only feasible moves"; a moderate penalty lets the search pass
        temporarily through infeasible solutions (as in modern hybrid GAs).
        """
        self.penalty = penalty
        # pad to K routes so that "unused vehicles" exist as empty routes
        self.routes = [list(r) for r in routes if r]
        while len(self.routes) < self.K:
            self.routes.append([])
        self.costs = [self._pcost(r) for r in self.routes]
        self._index()

        customers = list(range(1, self.inst.n_customers + 1))
        improved = True
        while improved:
            improved = False
            self.rng.shuffle(customers)
            for u in customers:
                if self._moves_for(u):
                    improved = True
            if self._reversals():
                improved = True
        return [r for r in self.routes if r]
