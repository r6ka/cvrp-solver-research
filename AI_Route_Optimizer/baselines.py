"""
baselines.py – simple non-AI methods, used only for COMPARISON

  sweep_fill        : sort customers by angle around the depot and fill each
                      truck until it is full (a typical "manual" dispatch plan)
  nearest_fill      : always drive to the nearest unserved customer that
                      still fits in the truck
  local_search_only : sweep_fill followed by ONE local search (no genetics)

These show how much the Genetic Algorithm adds on top of classic heuristics.
"""
import math

from local_search import LocalSearch


def sweep_fill(inst):
    x0, y0 = inst.coords[0]
    order = sorted(range(1, inst.n_customers + 1),
                   key=lambda c: math.atan2(inst.coords[c][1] - y0, inst.coords[c][0] - x0))
    routes, cur, load = [], [], 0.0
    for c in order:
        if load + inst.demand[c] > inst.capacity:
            routes.append(cur); cur, load = [], 0.0
        cur.append(c); load += inst.demand[c]
    if cur:
        routes.append(cur)
    return routes


def nearest_fill(inst):
    D, left, routes = inst.dist, set(range(1, inst.n_customers + 1)), []
    while left:
        cur, load, pos = [], 0.0, 0
        while True:
            cand = [c for c in left if load + inst.demand[c] <= inst.capacity]
            if not cand:
                break
            nxt = min(cand, key=lambda c: D[pos][c])
            cur.append(nxt); load += inst.demand[nxt]; left.remove(nxt); pos = nxt
        routes.append(cur)
    return routes


def local_search_only(inst, objective, seed=1):
    import random
    start = sweep_fill(inst)
    if len(start) > inst.num_vehicles:
        start = nearest_fill(inst)
    # allow as many routes as the start solution has (the baseline may need more trucks)
    k = inst.num_vehicles
    inst.num_vehicles = max(k, len(start))
    try:
        ls = LocalSearch(inst, objective, 12, random.Random(seed))
        return ls.run(start, penalty=1e9)
    finally:
        inst.num_vehicles = k
