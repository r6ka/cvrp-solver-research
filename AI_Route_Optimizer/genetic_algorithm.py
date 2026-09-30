"""
genetic_algorithm.py
--------------------
Hybrid (memetic) Genetic Algorithm for the CVRP.

Chromosome
    A "giant tour": a permutation of all customers, without depot visits.
    Example: [5, 3, 1, 7, 2, 4, 6]

Decoder (Split)
    Dynamic programming cuts the giant tour into at most K routes in the
    optimal way (Prins, 2004), e.g.  [5,3,1 | 7,2 | 4,6]
    -> Vehicle 1: 0-5-3-1-0, Vehicle 2: 0-7-2-0, Vehicle 3: 0-4-6-0

Fitness
    fitness = objective (distance or CO2) + penalty * capacity excess
    Lower is better. Only solutions with zero excess are reported.

One generation (steady state)
    1. Selection   – binary tournament picks two parents
    2. Crossover   – Order Crossover (OX) creates a child giant tour
    3. Mutation    – random swap / inversion with a small probability
    4. Decode      – Split the child into routes
    5. Education   – Local Search (2-opt, 2-opt*, relocate, swap, or-opt)
    6. Repair      – if still over capacity, re-run LS with 10x penalty
    7. Survival    – child joins the population, clones and worst removed
    The penalty adapts automatically to keep a healthy share of feasible
    children. If the search stagnates, the population is restarted while
    the best solution is kept.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from local_search import LocalSearch


@dataclass
class GAConfig:
    population_size: int = 25        # mu: survivors per generation
    generation_size: int = 40        # lambda: extra children before survivor selection
    mutation_rate: float = 0.2
    granularity: int = 12
    max_iterations: int = 20000      # children created
    max_no_improve: int = 4000       # stop after this many children without improvement
    restart_after: int = 1500        # restart population after this many without improvement
    time_limit: float = 120.0        # seconds
    target_feasible: float = 0.3     # desired share of feasible children
    seed: int = 1
    verbose: bool = True


@dataclass
class Individual:
    tour: list
    routes: list
    cost: float          # objective value (distance or CO2)
    excess: float        # total capacity excess
    fitness: float = 0.0

    @property
    def feasible(self) -> bool:
        return self.excess <= 1e-9


@dataclass
class GAResult:
    routes: list
    cost: float
    feasible: bool
    iterations: int
    runtime: float
    history: list = field(default_factory=list)   # [(iteration, seconds, best_cost)]
    first_feasible_cost: float | None = None


class HybridGA:
    def __init__(self, inst, objective, config: GAConfig | None = None):
        self.inst = inst
        self.obj = objective
        self.cfg = config or GAConfig()
        self.rng = random.Random(self.cfg.seed)
        self.ls = LocalSearch(inst, objective, self.cfg.granularity, self.rng)
        self.n = inst.n_customers
        self.K = inst.num_vehicles
        q, Q = inst.demand, inst.capacity
        # initial penalty: "average arc length per unit of demand"
        avg_d = sum(inst.dist[0][1:]) / self.n
        avg_q = max(1e-9, sum(q) / self.n)
        self.penalty = max(1.0, avg_d / avg_q) * (3.0 if objective.kind == "distance" else 1.0)
        if objective.kind == "co2":
            self.penalty *= objective.gp.fuel_full_l_per_km * objective.gp.co2_kg_per_litre
        self.feasible_log: list[bool] = []

    # ---------------------------------------------------------------- decoding
    def split(self, tour):
        """Optimal split of a giant tour into <= K routes (penalised capacity)."""
        n, K = len(tour), self.K
        q, Q = self.inst.demand, self.inst.capacity
        INF = math.inf
        max_load = 1.5 * Q
        V = [[INF] * (n + 1) for _ in range(K + 1)]
        P = [[-1] * (n + 1) for _ in range(K + 1)]
        V[0][0] = 0.0
        cost = self.obj.route_cost
        pen = self.penalty
        for k in range(1, K + 1):
            prev, cur, pk = V[k - 1], V[k], P[k]
            for i in range(n):
                if prev[i] == INF:
                    continue
                load = 0.0
                for j in range(i, n):
                    load += q[tour[j]]
                    if load > max_load and j > i:
                        break
                    c = cost(tour[i:j + 1])
                    if load > Q:
                        c += pen * (load - Q)
                    val = prev[i] + c
                    if val < cur[j + 1]:
                        cur[j + 1] = val
                        pk[j + 1] = i
        best_k = min(range(1, K + 1), key=lambda k: V[k][n])
        if V[best_k][n] == INF:
            # extremely overloaded tour: fall back to K roughly equal pieces
            size = math.ceil(n / K)
            return [tour[i:i + size] for i in range(0, n, size)]
        routes, j, k = [], n, best_k
        while j > 0:
            i = P[k][j]
            routes.append(tour[i:j])
            j, k = i, k - 1
        return routes[::-1]

    # -------------------------------------------------------------- evaluation
    def _make(self, routes) -> Individual:
        q, Q = self.inst.demand, self.inst.capacity
        routes = [r for r in routes if r]
        cost = sum(self.obj.route_cost(r) for r in routes)
        excess = sum(max(0.0, sum(q[c] for c in r) - Q) for r in routes)
        tour = [c for r in routes for c in r]
        return Individual(tour, routes, cost, excess)

    def _fitness(self, ind: Individual) -> float:
        return ind.cost + self.penalty * ind.excess

    def educate(self, tour) -> Individual:
        routes = self.split(tour)
        routes = self.ls.run(routes, self.penalty)
        ind = self._make(routes)
        self.feasible_log.append(ind.feasible)
        if not ind.feasible:                      # repair step
            routes = self.ls.run(routes, self.penalty * 10)
            rep = self._make(routes)
            if not rep.feasible:
                routes = self.ls.run(routes, self.penalty * 100)
                rep = self._make(routes)
            if rep.feasible:
                ind = rep
        return ind

    # --------------------------------------------------------------- operators
    def order_crossover(self, p1, p2):
        n = len(p1)
        a, b = sorted(self.rng.sample(range(n), 2))
        child = [None] * n
        child[a:b + 1] = p1[a:b + 1]
        used = set(p1[a:b + 1])
        pos = (b + 1) % n
        for k in range(n):
            g = p2[(b + 1 + k) % n]
            if g not in used:
                child[pos] = g
                pos = (pos + 1) % n
        return child

    def mutate(self, tour):
        t = tour[:]
        if self.rng.random() < 0.5:
            i, j = self.rng.sample(range(len(t)), 2)
            t[i], t[j] = t[j], t[i]
        else:
            i, j = sorted(self.rng.sample(range(len(t)), 2))
            t[i:j + 1] = t[i:j + 1][::-1]
        return t

    def tournament(self, pop):
        a, b = self.rng.sample(pop, 2)
        return a if a.fitness <= b.fitness else b

    # --------------------------------------------------------- initialisation
    def _initial_tours(self, count):
        coords = self.inst.coords
        x0, y0 = coords[0]
        customers = list(range(1, self.n + 1))
        tours = []
        # sweep: sort customers by polar angle around the depot
        sweep = sorted(customers, key=lambda c: math.atan2(coords[c][1] - y0, coords[c][0] - x0))
        tours.append(sweep)
        # nearest neighbour tour
        D = self.inst.dist
        left, cur, nn = set(customers), 0, []
        while left:
            nxt = min(left, key=lambda c: D[cur][c])
            nn.append(nxt); left.remove(nxt); cur = nxt
        tours.append(nn)
        while len(tours) < count:
            t = customers[:]
            self.rng.shuffle(t)
            if self.rng.random() < 0.3:              # rotated sweep variant
                s = self.rng.randrange(self.n)
                t = sweep[s:] + sweep[:s]
            tours.append(t)
        return tours

    def _new_population(self, on_new=None):
        pop = []
        for t in self._initial_tours(self.cfg.population_size * 2):
            ind = self.educate(t)
            if on_new:
                on_new(ind)
            pop.append(ind)
        for ind in pop:
            ind.fitness = self._fitness(ind)
        return self._survivors(pop)

    def _survivors(self, pop):
        for ind in pop:
            ind.fitness = self._fitness(ind)
        pop.sort(key=lambda i: i.fitness)
        unique, seen = [], set()
        for ind in pop:                          # remove clones (diversity)
            key = round(ind.fitness, 6)
            if key in seen:
                continue
            seen.add(key)
            unique.append(ind)
        return unique[: self.cfg.population_size]

    def _adapt_penalty(self):
        if len(self.feasible_log) >= 100:
            share = sum(self.feasible_log[-100:]) / 100
            if share < self.cfg.target_feasible - 0.05:
                self.penalty *= 1.2
            elif share > self.cfg.target_feasible + 0.05:
                self.penalty *= 0.85
            self.feasible_log.clear()

    # --------------------------------------------------------------------- run
    def run(self) -> GAResult:
        cfg = self.cfg
        t0 = time.time()
        best: Individual | None = None
        first_feasible = None
        history = []

        def consider(ind, it):
            nonlocal best, first_feasible
            if ind.feasible and (best is None or ind.cost < best.cost - 1e-9):
                best = ind
                if first_feasible is None:
                    first_feasible = ind.cost
                history.append((it, round(time.time() - t0, 3), ind.cost))
                if cfg.verbose:
                    print(f"  iter {it:6d}  t={time.time()-t0:6.1f}s  best={ind.cost:.2f}  "
                          f"routes={len(ind.routes)}", flush=True)
                return True
            return False

        pop = self._new_population(lambda ind: consider(ind, 0))

        it, last_improve, last_restart = 0, 0, 0
        while it < cfg.max_iterations:
            if time.time() - t0 > cfg.time_limit:
                break
            if it - last_improve > cfg.max_no_improve:
                break
            it += 1
            p1, p2 = self.tournament(pop), self.tournament(pop)
            child = self.order_crossover(p1.tour, p2.tour)
            if self.rng.random() < cfg.mutation_rate:
                child = self.mutate(child)
            ind = self.educate(child)
            ind.fitness = self._fitness(ind)
            if consider(ind, it):
                last_improve = it
            pop.append(ind)
            if len(pop) >= cfg.population_size + cfg.generation_size:
                pop = self._survivors(pop)
            if it % 100 == 0:
                self._adapt_penalty()
            if it - max(last_improve, last_restart) > cfg.restart_after:
                if cfg.verbose:
                    print(f"  iter {it:6d}  restart population (diversification)", flush=True)
                pop = self._new_population(lambda ind: consider(ind, it))
                if best is not None:
                    pop.append(best)
                last_restart = it

        runtime = time.time() - t0
        if best is None:  # no feasible solution found: return the least-bad one
            b = min(pop, key=lambda i: i.excess)
            return GAResult(b.routes, b.cost, False, it, runtime, history, None)
        return GAResult(best.routes, best.cost, True, it, runtime, history, first_feasible)
