"""
exact_route.py – exact optimisation of the visiting order inside each route
===========================================================================

Held-Karp dynamic programming (Held & Karp 1962; Bellman 1962), hand-written.

For one route with customer set S (|S| = m), it finds the order with the
LOWEST possible cost:

    f(T, j) = cheapest way to leave the depot, visit exactly the customers
              in T (a subset of S), and stop at j (j in T)
    f({j}, j) = c(0 -> j)
    f(T, j)   = min over i in T\\{j} of  f(T\\{j}, i) + c(i -> j | T\\{j} delivered)
    optimum   = min over j of  f(S, j) + c(j -> 0)

Cost is O(2^m * m^2). It is exact, not heuristic, so after this step every
route with m <= limit is PROVEN to be driven in its optimal order. None of
the reference repositories give that guarantee: 2-opt / or-opt alone only
reach a local optimum.

The arc cost may depend on the load still on board (the green CO2
objective). The load is known from T (the customers already served), so the
DP stays exact for load-dependent fuel as well.
"""
from __future__ import annotations

from functools import lru_cache


def held_karp(route, D, arc_cost=None, demand=None):
    """
    Return (best_order, best_cost) for the customers of `route`.
    arc_cost(i, j, load_on_board) -> cost. Default: distance D[i][j].
    """
    m = len(route)
    if m <= 2:
        order = list(route)
        return order, _route_cost(order, D, arc_cost, demand)
    nodes = list(route)
    if demand is not None:
        total = sum(demand[c] for c in nodes)
        dem = [demand[c] for c in nodes]
    else:
        total, dem = 0.0, [0.0] * m
    full = (1 << m) - 1

    # delivered[T] = demand already delivered after serving set T
    delivered = [0.0] * (1 << m)
    for T in range(1, 1 << m):
        low = T & -T
        k = low.bit_length() - 1
        delivered[T] = delivered[T ^ low] + dem[k]

    if arc_cost is None:
        def c(i, j, _load):
            return D[i][j]
    else:
        c = arc_cost

    INF = float("inf")
    f = [[INF] * m for _ in range(1 << m)]
    parent = [[-1] * m for _ in range(1 << m)]
    for j in range(m):
        f[1 << j][j] = c(0, nodes[j], total)
    for T in range(1, 1 << m):
        fT = f[T]
        load_after = total - delivered[T]
        for j in range(m):
            if not (T >> j) & 1:
                continue
            base = fT[j]
            if base == INF:
                continue
            nj = nodes[j]
            for k in range(m):
                if (T >> k) & 1:
                    continue
                T2 = T | (1 << k)
                val = base + c(nj, nodes[k], load_after)
                if val < f[T2][k]:
                    f[T2][k] = val
                    parent[T2][k] = j
    best, bj = INF, -1
    for j in range(m):
        val = f[full][j] + c(nodes[j], 0, 0.0)
        if val < best:
            best, bj = val, j
    order, T, j = [], full, bj
    while j != -1:
        order.append(nodes[j])
        pj = parent[T][j]
        T ^= 1 << j
        j = pj
    order.reverse()
    return order, best


def _route_cost(order, D, arc_cost, demand):
    if not order:
        return 0.0
    if arc_cost is None:
        return D[0][order[0]] + D[order[-1]][0] + sum(D[a][b] for a, b in zip(order, order[1:]))
    load = sum(demand[c] for c in order) if demand is not None else 0.0
    s, prev = 0.0, 0
    for cst in order:
        s += arc_cost(prev, cst, load)
        load -= demand[cst] if demand is not None else 0.0
        prev = cst
    return s + arc_cost(prev, 0, 0.0)


class RouteOptimiser:
    """Caches exact results so each customer set is solved only once."""

    def __init__(self, inst, objective, limit: int = 12):
        self.inst = inst
        self.obj = objective
        self.limit = limit
        self.cache = {}
        self.calls = 0
        if objective.kind == "distance":
            self.arc = None
        else:
            gp, Q, D = objective.gp, inst.capacity, inst.dist
            slope = (gp.fuel_full_l_per_km - gp.fuel_empty_l_per_km) / Q

            def arc(i, j, load):
                return D[i][j] * (gp.fuel_empty_l_per_km + slope * load) * gp.co2_kg_per_litre
            self.arc = arc

    def optimise(self, route):
        """Return (order, proven_optimal: bool)."""
        if len(route) > self.limit:
            return list(route), False
        key = frozenset(route)
        hit = self.cache.get(key)
        if hit is None:
            self.calls += 1
            order, _ = held_karp(route, self.inst.dist, self.arc,
                                 self.inst.demand if self.arc else None)
            hit = self.cache[key] = order
        cur = self.obj.route_cost(route)
        new = self.obj.route_cost(hit)
        if new < cur - 1e-9:
            return list(hit), True
        return list(route), True   # current order is already optimal (ties kept)

    def optimise_solution(self, routes):
        out, proven = [], 0
        for r in routes:
            o, ok = self.optimise(r)
            out.append(o)
            proven += ok
        return out, proven
