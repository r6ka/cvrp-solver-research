"""
classic_methods.py – our re-implementations of the methods used in the
"manual implementation" reference repositories (for COMPARISON only)
======================================================================

Each function follows the step list documented for that repository in
CVRP-Manual-Implementation-Reference.pdf. The code is written by us from
those descriptions (it is NOT the original repositories' code), so all
methods run on the same parser, distance matrix and time limit.

  cws_parallel      VeRyPy "ps", kavetinaveen/CWVRP, adamiaonr (CWS step)
  cws_sequential    VeRyPy "ss", kavetinaveen/CWVRP (sequential)
  sweep_only        ivanchoff/TS_CVRP phase 1, VeRyPy "swp"
  greedy_nn         vss2sn/cvrp Algorithm 1 (greedy)
  rfcs              msommacal/cvrp route-first cluster-second (Beasley 1983)
  cws_2opt          MekhyW 2-opt applied to each CWS route
  simulated_annealing   vss2sn/cvrp Algorithm 6 (random 2-vehicle move,
                        geometric cooling, reheats) / adamiaonr CWS + SA
  tabu_search       vss2sn Algorithm 4 / ivanchoff / AHalic: relocate
                    neighbourhood, tabu tenure, aspiration
  simple_ga         manu-p-1/cvrp: OX crossover (0.85), swap/inversion
                    mutation (0.15), tournament 5, greedy capacity split,
                    NO local search
"""
from __future__ import annotations

import math
import random
import time

from construction import clarke_wright_parallel, clarke_wright_sequential, nearest_neighbour, sweep


def _rc(r, D):
    if not r:
        return 0.0
    return D[0][r[0]] + D[r[-1]][0] + sum(D[a][b] for a, b in zip(r, r[1:]))


def cws_parallel(inst, **_):
    return clarke_wright_parallel(inst)


def cws_sequential(inst, **_):
    return clarke_wright_sequential(inst)


def sweep_only(inst, **_):
    return sweep(inst)


def greedy_nn(inst, **_):
    return nearest_neighbour(inst)


def two_opt_route(r, D):
    r = r[:]
    improved = True
    while improved:
        improved = False
        seq = [0] + r + [0]
        for i in range(1, len(seq) - 2):
            for j in range(i + 1, len(seq) - 1):
                d = D[seq[i - 1]][seq[j]] + D[seq[i]][seq[j + 1]] - D[seq[i - 1]][seq[i]] - D[seq[j]][seq[j + 1]]
                if d < -1e-9:
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    improved = True
        r = seq[1:-1]
    return r


def cws_2opt(inst, **_):
    return [two_opt_route(r, inst.dist) for r in clarke_wright_parallel(inst)]


def rfcs(inst, **_):
    """Route first (nearest-neighbour giant tour), cluster second (optimal split, unlimited fleet)."""
    D, q, Q, n = inst.dist, inst.demand, inst.capacity, inst.n_customers
    left, tour, cur = set(range(1, n + 1)), [], 0
    while left:
        nxt = min(left, key=lambda c: D[cur][c])
        tour.append(nxt); left.remove(nxt); cur = nxt
    INF = float("inf")
    V, P = [0.0] + [INF] * n, [-1] * (n + 1)
    for i in range(n):
        load, inner = 0.0, 0.0
        for j in range(i, n):
            load += q[tour[j]]
            if load > Q:
                break
            if j > i:
                inner += D[tour[j - 1]][tour[j]]
            c = D[0][tour[i]] + inner + D[tour[j]][0]
            if V[i] + c < V[j + 1]:
                V[j + 1], P[j + 1] = V[i] + c, i
    routes, j = [], n
    while j > 0:
        routes.append(tour[P[j]:j]); j = P[j]
    return routes[::-1]


def simulated_annealing(inst, time_limit=10.0, seed=1, **_):
    """vss2sn SA: move a random node of one random vehicle after a random node of
    another; Metropolis acceptance; geometric cooling; reheat on stagnation."""
    rng = random.Random(seed)
    D, q, Q = inst.dist, inst.demand, inst.capacity
    routes = [r[:] for r in clarke_wright_parallel(inst)]   # adamiaonr: SA on top of CWS
    loads = [sum(q[c] for c in r) for r in routes]
    cost = sum(_rc(r, D) for r in routes)
    best, best_cost = [r[:] for r in routes], cost
    T0 = max(1.0, 0.05 * cost / max(1, inst.n_customers))
    T, alpha, stag, stag_limit = T0, 0.9995, 0, 5000
    t0 = time.time()
    while time.time() - t0 < time_limit:
        a = rng.randrange(len(routes))
        b = rng.randrange(len(routes))
        if not routes[a]:
            continue
        i = rng.randrange(len(routes[a]))
        u = routes[a][i]
        if a != b and loads[b] + q[u] > Q:
            continue
        A = routes[a][:i] + routes[a][i + 1:]
        B = A if a == b else routes[b]
        j = rng.randrange(len(B) + 1)
        B2 = B[:j] + [u] + B[j:]
        if a == b:
            delta = _rc(B2, D) - _rc(routes[a], D)
        else:
            delta = _rc(A, D) + _rc(B2, D) - _rc(routes[a], D) - _rc(routes[b], D)
        if delta < 0 or rng.random() < math.exp(-delta / T):
            if a == b:
                routes[a] = B2
            else:
                routes[a], routes[b] = A, B2
                loads[a] -= q[u]; loads[b] += q[u]
            cost += delta
            if cost < best_cost - 1e-9:
                best, best_cost, stag = [r[:] for r in routes], cost, 0
            else:
                stag += 1
        else:
            stag += 1
        T *= alpha
        if stag > stag_limit:          # reheat
            T, stag = T0, 0
    return [r for r in best if r]


def tabu_search(inst, time_limit=10.0, seed=1, tenure=10, **_):
    """Relocate neighbourhood; best non-tabu move each iteration (may worsen);
    aspiration if it gives a new best; tabu = (customer, origin route)."""
    rng = random.Random(seed)
    D, q, Q = inst.dist, inst.demand, inst.capacity
    routes = [r[:] for r in sweep(inst)]
    while len(routes) < inst.num_vehicles:
        routes.append([])
    loads = [sum(q[c] for c in r) for r in routes]
    cost = sum(_rc(r, D) for r in routes)
    best, best_cost = [r[:] for r in routes], cost
    tabu = {}
    it = 0
    t0 = time.time()
    while time.time() - t0 < time_limit:
        it += 1
        cand = None
        for a, A in enumerate(routes):
            for i, u in enumerate(A):
                pu = A[i - 1] if i else 0
                xu = A[i + 1] if i + 1 < len(A) else 0
                rem = D[pu][xu] - D[pu][u] - D[u][xu]
                for b, B in enumerate(routes):
                    if b == a or loads[b] + q[u] > Q:
                        continue
                    prev = 0
                    for j, nx in enumerate(B + [0]):
                        d = rem + D[prev][u] + D[u][nx] - D[prev][nx]
                        is_tabu = tabu.get((u, b), 0) > it
                        if (not is_tabu or cost + d < best_cost - 1e-9) and \
                                (cand is None or d < cand[0] or (d == cand[0] and rng.random() < 0.5)):
                            cand = (d, a, i, b, j)
                        prev = nx
        if cand is None:
            break
        d, a, i, b, j = cand
        u = routes[a].pop(i)
        routes[b].insert(j, u)
        loads[a] -= q[u]; loads[b] += q[u]
        cost += d
        tabu[(u, a)] = it + tenure + rng.randint(0, 5)
        if cost < best_cost - 1e-9:
            best, best_cost = [r[:] for r in routes], cost
    return [r for r in best if r]


def simple_ga(inst, time_limit=10.0, seed=1, pop_size=200, **_):
    """manu-p-1 style GA without local search."""
    rng = random.Random(seed)
    D, q, Q, n = inst.dist, inst.demand, inst.capacity, inst.n_customers

    def decode(t):
        routes, cur, ld = [], [], 0.0
        for c in t:
            if ld + q[c] > Q:
                routes.append(cur); cur, ld = [], 0.0
            cur.append(c); ld += q[c]
        routes.append(cur)
        return routes

    def fit(t):
        rs = decode(t)
        extra = max(0, len(rs) - inst.num_vehicles) * 1000.0   # fleet violation penalty
        return sum(_rc(r, D) for r in rs) + extra

    def ox(p1, p2):
        a, b = sorted(rng.sample(range(n), 2))
        child = [None] * n
        child[a:b + 1] = p1[a:b + 1]
        used = set(p1[a:b + 1])
        fill = [g for g in p2 if g not in used]
        k = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[k]; k += 1
        return child

    base = list(range(1, n + 1))
    pop = []
    for _ in range(pop_size):
        t = base[:]; rng.shuffle(t); pop.append((fit(t), t))
    t0 = time.time()
    while time.time() - t0 < time_limit:
        def tour5():
            return min(rng.sample(pop, 5), key=lambda x: x[0])[1]
        p1, p2 = tour5(), tour5()
        c = ox(p1, p2) if rng.random() < 0.85 else p1[:]
        if rng.random() < 0.15:
            i, j = sorted(rng.sample(range(n), 2))
            if rng.random() < 0.5:
                c[i], c[j] = c[j], c[i]
            else:
                c[i:j + 1] = c[i:j + 1][::-1]
        f = fit(c)
        worst = max(range(len(pop)), key=lambda k: pop[k][0])
        if f < pop[worst][0]:
            pop[worst] = (f, c)
    best = min(pop, key=lambda x: x[0])[1]
    return decode(best)


METHODS = [
    ("CWS parallel (VeRyPy ps / CWVRP / adamiaonr)", cws_parallel, False),
    ("CWS sequential (VeRyPy ss / CWVRP)", cws_sequential, False),
    ("Sweep (ivanchoff / VeRyPy swp)", sweep_only, False),
    ("Greedy NN (vss2sn Alg. 1)", greedy_nn, False),
    ("Route-first cluster-second (msommacal)", rfcs, False),
    ("CWS + 2-opt (MekhyW)", cws_2opt, False),
    ("CWS + Simulated annealing (vss2sn / adamiaonr)", simulated_annealing, True),
    ("Tabu search (vss2sn / ivanchoff / AHalic)", tabu_search, True),
    ("Simple GA, no LS (manu-p-1)", simple_ga, True),
]
