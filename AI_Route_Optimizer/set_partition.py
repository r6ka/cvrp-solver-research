"""
set_partition.py – exact recombination of good routes (hand-written branch & bound)
===================================================================================

While the GA runs, every route of every good local optimum goes into a ROUTE
POOL (only the cheapest order found for each customer set is kept). At the end
we solve, exactly, the set-partitioning problem over that pool:

    choose routes from the pool so that every customer is covered EXACTLY once,
    at most K routes are used, and the total cost is minimal.

This often finds a better solution than any single GA individual, because it
can mix routes that were never in the same parent. The HGS / GA / tabu repos
in the reference list do not do this. It is used in state-of-the-art hybrids
such as LKH-SP / FILO2 + SP, but those call a commercial or LP solver. Here it
is a small branch & bound written by hand:

  Branching  pick the uncovered customer with the FEWEST candidate routes and
             try each compatible route containing it (cheapest share first)
  Bound 1    cost so far + sum over uncovered customers of their minimum
             "demand share" of a route cost:
                 share(r, c) = cost(r) * q_c / load(r)
             This is a valid lower bound: the routes of any completion split
             their cost over their customers in exactly this way.
  Bound 2    routes used + ceil(remaining demand / Q) <= K
  Anytime    it starts from the GA's best cost as the upper bound and stops at
             the time limit, keeping the best complete cover found.
"""
from __future__ import annotations

import math
import time


class RoutePool:
    def __init__(self, inst, objective, max_size: int = 60000):
        self.inst = inst
        self.obj = objective
        self.routes = {}          # frozenset -> (cost, order)
        self.max_size = max_size

    def add(self, route):
        if not route:
            return
        q, Q = self.inst.demand, self.inst.capacity
        if sum(q[c] for c in route) > Q + 1e-9:
            return
        key = frozenset(route)
        cost = self.obj.route_cost(route)
        old = self.routes.get(key)
        if old is None or cost < old[0] - 1e-9:
            if old is None and len(self.routes) >= self.max_size:
                return
            self.routes[key] = (cost, list(route))

    def add_solution(self, routes):
        for r in routes:
            self.add(r)

    def __len__(self):
        return len(self.routes)


def solve_set_partitioning(pool: RoutePool, upper_bound: float, time_limit: float = 10.0,
                           tracer=None):
    inst = pool.inst
    n, K, Q = inst.n_customers, inst.num_vehicles, inst.capacity
    q = inst.demand
    t0 = time.time()

    items = []   # (cost, mask, order, load)
    for key, (cost, order) in pool.routes.items():
        mask = 0
        for c in key:
            mask |= 1 << c
        items.append((cost, mask, order, sum(q[c] for c in key)))

    # minimum demand-share of each customer (bound 1)
    share_min = [0.0] * (n + 1)
    cover = [[] for _ in range(n + 1)]
    for idx, (cost, mask, order, load) in enumerate(items):
        for c in order:
            w = (q[c] / load) if load > 0 else 1.0 / len(order)
            s = cost * w
            cover[c].append((s, idx))
    for c in range(1, n + 1):
        if not cover[c]:
            return None, {"status": "some customer has no route in the pool"}
        cover[c].sort()
        share_min[c] = cover[c][0][0]

    full = 1          # bit 0 = depot (always "covered")
    for c in range(1, n + 1):
        full |= 1 << c
    total_demand = sum(q[1:])

    best = {"cost": upper_bound, "sel": None}
    stats = {"nodes": 0, "pruned": 0, "timeout": False}

    def dfs(covered, cost, lb_rest, used, dem_left, sel):
        stats["nodes"] += 1
        if stats["nodes"] % 2000 == 0 and time.time() - t0 > time_limit:
            stats["timeout"] = True
            return
        if covered == full:
            if cost < best["cost"] - 1e-9:
                best["cost"], best["sel"] = cost, list(sel)
                if tracer:
                    tracer.log(f"   SP: new best cover {cost:.2f} with {len(sel)} routes "
                               f"(after {stats['nodes']} nodes)", force=True)
            return
        if cost + lb_rest >= best["cost"] - 1e-9:
            stats["pruned"] += 1
            return
        if used + math.ceil(dem_left / Q - 1e-9) > K:
            stats["pruned"] += 1
            return
        # branch on the uncovered customer with the fewest compatible routes
        best_c, best_list = None, None
        for c in range(1, n + 1):
            if (covered >> c) & 1:
                continue
            lst = [idx for _, idx in cover[c] if not (items[idx][1] & covered)]
            if best_list is None or len(lst) < len(best_list):
                best_c, best_list = c, lst
                if len(lst) <= 1:
                    break
        if not best_list:
            return
        for idx in best_list:
            cost_r, mask, order, load = items[idx]
            # new lower bound for the rest
            removed = 0.0
            m = mask
            while m:
                low = m & -m
                c = low.bit_length() - 1
                removed += share_min[c]
                m ^= low
            sel.append(idx)
            dfs(covered | mask, cost + cost_r, lb_rest - removed, used + 1, dem_left - load, sel)
            sel.pop()
            if stats["timeout"]:
                return

    lb0 = sum(share_min[1:])
    dfs(1, 0.0, lb0, 0, total_demand, [])   # bit 0 = depot, marked as covered
    info = {
        "pool_routes": len(items),
        "nodes": stats["nodes"],
        "pruned": stats["pruned"],
        "proven_optimal_over_pool": not stats["timeout"],
        "root_lower_bound": lb0,
        "seconds": round(time.time() - t0, 2),
    }
    if best["sel"] is None:
        return None, info
    routes = [items[i][2] for i in best["sel"]]
    return routes, info
