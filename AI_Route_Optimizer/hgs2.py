"""
hgs2.py – v2.0 solver engine (all hand-written, standard library only)
=====================================================================

    1. CONSTRUCT  12+ hand-written heuristics (construction.py) + random tours
    2. EVOLVE     hybrid genetic search:
                    * giant-tour chromosome, Order Crossover (OX)
                    * Split decoder (Bellman shortest path, fleet limit K)
                    * FastLocalSearch with O(1) move evaluation + SWAP*
                    * feasible / infeasible sub-populations
                    * biased fitness = cost rank + diversity rank
                      (broken-pairs distance, Vidal et al. 2012)
                    * adaptive capacity penalty (target 20% feasible)
    3. LEARN      UCB1 multi-armed bandit chooses the mutation
                  (none / random ruin / proximity ruin / route ruin + regret-2
                  repair) from the rewards it has seen during THIS run
    4. EXACT      every new best: Held-Karp makes each route order optimal
    5. RECOMBINE  exact set partitioning over the pool of all good routes
    6. REPORT     trace of every step, move statistics, bandit statistics

What the reference repositories lack, and how we handle it:
    fleet limit ignored (CWS, sweep)       -> Split with at most K routes + penalty repair
    GA may output infeasible (MakeValid)   -> only zero-excess solutions are reported
    local optimum only (2-opt, tabu, SA)   -> Held-Karp proves each route order optimal
    no mixing of routes across solutions   -> exact set partitioning over the route pool
    O(route) move evaluation (v1, most)    -> O(1) hand-derived deltas (tests/test_moves.py)
    no step-by-step evidence               -> --trace writes every construction step,
                                              every new best, bandit statistics
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from construction import Tracer, construction_portfolio
from exact_route import RouteOptimiser
from local_search import LocalSearch
from local_search2 import FastLocalSearch
from set_partition import RoutePool, solve_set_partitioning


@dataclass
class HGS2Config:
    mu: int = 25                  # minimum sub-population size
    lam: int = 40                 # generation size
    n_elite: int = 5
    n_close: int = 4
    granularity: int = 20
    target_feasible: float = 0.2
    repair_prob: float = 0.5
    time_limit: float = 60.0
    max_no_improve: int = 20000
    restart_after: int = 4000
    exact_route_limit: int = 12
    sp_time: float = 10.0         # seconds for the final set-partitioning step
    use_bandit: bool = True
    use_swap_star: bool = True
    use_exact: bool = True
    use_sp: bool = True
    seed: int = 1
    verbose: bool = True


@dataclass
class Indiv:
    routes: list
    cost: float
    excess: float
    tour: list
    succ: list
    pred: list
    pcost: float = 0.0
    prox: list = field(default_factory=list)   # sorted [(distance, Indiv)]
    biased: float = 0.0

    @property
    def feasible(self):
        return self.excess <= 1e-9


@dataclass
class Result2:
    routes: list
    cost: float
    feasible: bool
    iterations: int
    runtime: float
    history: list
    first_feasible_cost: float | None
    stats: dict


# ============================================================ helpers
def broken_pairs(a: Indiv, b: Indiv, n: int) -> float:
    diff = 0
    sa, pa, sb, pb = a.succ, a.pred, b.succ, b.pred
    for j in range(1, n + 1):
        if sa[j] != sb[j] and sa[j] != pb[j]:
            diff += 1
        if pa[j] == 0 and pb[j] != 0 and sb[j] != 0:
            diff += 1
    return diff / n


class SubPop:
    def __init__(self, cfg, n):
        self.cfg, self.n, self.inds = cfg, n, []

    def add(self, ind):
        for o in self.inds:
            d = broken_pairs(ind, o, self.n)
            _insort(ind.prox, (d, id(o), o))
            _insort(o.prox, (d, id(ind), ind))
        self.inds.append(ind)
        if len(self.inds) >= self.cfg.mu + self.cfg.lam:
            while len(self.inds) > self.cfg.mu:
                self._remove_worst()

    def update_biased(self):
        m = len(self.inds)
        if m == 0:
            return
        if m == 1:
            self.inds[0].biased = 0.0
            return
        by_cost = sorted(self.inds, key=lambda i: i.pcost)
        div = {id(i): self._avg_close(i) for i in self.inds}
        by_div = sorted(self.inds, key=lambda i: -div[id(i)])
        r_cost = {id(i): k / (m - 1) for k, i in enumerate(by_cost)}
        r_div = {id(i): k / (m - 1) for k, i in enumerate(by_div)}
        w = 1.0 - self.cfg.n_elite / m
        for i in self.inds:
            i.biased = r_cost[id(i)] + w * r_div[id(i)]

    def _avg_close(self, ind):
        k = min(self.cfg.n_close, len(ind.prox))
        if k == 0:
            return 0.0
        return sum(d for d, _, _ in ind.prox[:k]) / k

    def _remove_worst(self):
        self.update_biased()
        worst, worst_clone = None, False
        for i in self.inds:
            clone = bool(i.prox) and i.prox[0][0] < 1e-9
            if worst is None or (clone and not worst_clone) or \
                    (clone == worst_clone and i.biased > worst.biased):
                worst, worst_clone = i, clone
        self.inds.remove(worst)
        for o in self.inds:
            o.prox = [t for t in o.prox if t[2] is not worst]
        worst.prox = []


def _insort(lst, item):
    lo, hi = 0, len(lst)
    while lo < hi:
        mid = (lo + hi) // 2
        if lst[mid][0] < item[0]:
            lo = mid + 1
        else:
            hi = mid
    lst.insert(lo, item)


class UCB1:
    """Multi-armed bandit (Auer et al. 2002). Learns which mutation pays off."""

    def __init__(self, arms, rng, c=0.5):
        self.arms, self.rng, self.c = arms, rng, c
        self.n = {a: 0 for a in arms}
        self.s = {a: 0.0 for a in arms}
        self.t = 0

    def select(self):
        self.t += 1
        for a in self.arms:
            if self.n[a] == 0:
                return a
        return max(self.arms, key=lambda a: self.s[a] / self.n[a]
                   + self.c * math.sqrt(2 * math.log(self.t) / self.n[a]))

    def update(self, arm, reward):
        self.n[arm] += 1
        self.s[arm] += reward

    def table(self):
        return {a: {"times_used": self.n[a],
                    "mean_reward": round(self.s[a] / self.n[a], 4) if self.n[a] else 0.0}
                for a in self.arms}


# ============================================================ solver
class HGS2:
    ARMS = ["crossover only", "random ruin", "proximity ruin", "route ruin"]

    def __init__(self, inst, objective, cfg: HGS2Config | None = None, tracer: Tracer | None = None):
        self.inst, self.obj = inst, objective
        self.cfg = cfg or HGS2Config()
        self.tr = tracer or Tracer(False)
        self.rng = random.Random(self.cfg.seed)
        self.n, self.K = inst.n_customers, inst.num_vehicles
        self.D, self.q, self.Q = inst.dist, inst.demand, inst.capacity
        symmetric = all(abs(self.D[i][j] - self.D[j][i]) < 1e-9
                        for i in range(0, self.n + 1, max(1, self.n // 20)) for j in range(self.n + 1))
        self.fast = objective.kind == "distance" and symmetric
        if self.fast:
            self.ls = FastLocalSearch(inst, self.cfg.granularity, self.rng)
        else:
            self.ls = LocalSearch(inst, objective, min(self.cfg.granularity, 12), self.rng)
        self.exact = RouteOptimiser(inst, objective, self.cfg.exact_route_limit)
        self.pool = RoutePool(inst, objective)
        self.bandit = UCB1(self.ARMS, self.rng)
        avg_d = sum(self.D[0][1:]) / self.n
        avg_q = max(1e-9, sum(self.q) / self.n)
        self.penalty = max(1.0, min(1000.0, avg_d / avg_q))
        if objective.kind == "co2":
            self.penalty *= objective.gp.fuel_full_l_per_km * objective.gp.co2_kg_per_litre
        self.feas_log = []
        self.neigh = [[]] + [
            [v for _, v in sorted((self.D[u][v], v) for v in range(1, self.n + 1) if v != u)[:30]]
            for u in range(1, self.n + 1)]

    # ------------------------------------------------------------- basics
    def route_cost(self, r):
        return self.obj.route_cost(r)

    def make(self, routes) -> Indiv:
        routes = [list(r) for r in routes if r]
        q, Q = self.q, self.Q
        cost = sum(self.route_cost(r) for r in routes)
        excess = sum(max(0.0, sum(q[c] for c in r) - Q) for r in routes)
        succ = [0] * (self.n + 1)
        pred = [0] * (self.n + 1)
        for r in routes:
            for k, c in enumerate(r):
                pred[c] = r[k - 1] if k > 0 else 0
                succ[c] = r[k + 1] if k + 1 < len(r) else 0
        ind = Indiv(routes, cost, excess, [c for r in routes for c in r], succ, pred)
        ind.pcost = cost + self.penalty * excess
        return ind

    def split(self, tour):
        """Bellman split: first unlimited fleet (O(n*L)), then K-limited if needed."""
        n, K = len(tour), self.K
        q, Q, D = self.q, self.Q, self.D
        pen = self.penalty
        fast = self.fast
        max_load = 1.5 * Q
        INF = float("inf")

        def seg_costs(i):
            load, inner = 0.0, 0.0
            for j in range(i, n):
                c = tour[j]
                load += q[c]
                if j > i:
                    if load > max_load:
                        return
                    inner += D[tour[j - 1]][c] if fast else 0.0
                if fast:
                    cost = D[0][tour[i]] + inner + D[c][0]
                else:
                    cost = self.route_cost(tour[i:j + 1])
                if load > Q:
                    cost += pen * (load - Q)
                yield j, cost

        # unlimited fleet
        V = [INF] * (n + 1)
        P = [-1] * (n + 1)
        V[0] = 0.0
        for i in range(n):
            if V[i] == INF:
                continue
            for j, c in seg_costs(i):
                if V[i] + c < V[j + 1]:
                    V[j + 1] = V[i] + c
                    P[j + 1] = i
        routes, j = [], n
        while j > 0:
            i = P[j]
            routes.append(tour[i:j])
            j = i
        if len(routes) <= K:
            return routes[::-1]
        # K-limited version
        Vk = [[INF] * (n + 1) for _ in range(K + 1)]
        Pk = [[-1] * (n + 1) for _ in range(K + 1)]
        Vk[0][0] = 0.0
        for k in range(1, K + 1):
            prev, cur, pk = Vk[k - 1], Vk[k], Pk[k]
            for i in range(n):
                if prev[i] == INF:
                    continue
                for j, c in seg_costs(i):
                    if prev[i] + c < cur[j + 1]:
                        cur[j + 1] = prev[i] + c
                        pk[j + 1] = i
        if Vk[K][n] == INF:
            size = math.ceil(n / K)
            return [tour[i:i + size] for i in range(0, n, size)]
        best_k = min(range(1, K + 1), key=lambda k: Vk[k][n])
        routes, j, k = [], n, best_k
        while j > 0:
            i = Pk[k][j]
            routes.append(tour[i:j])
            j, k = i, k - 1
        return routes[::-1]

    # ----------------------------------------------------- ruin & recreate
    def ruin(self, routes, arm):
        n = self.n
        rng = self.rng
        k = rng.randint(max(2, n // 20), max(3, n // 7))
        removed = set()
        if arm == "random ruin":
            removed = set(rng.sample(range(1, n + 1), min(k, n)))
        elif arm == "proximity ruin":
            seed = rng.randint(1, n)
            removed = {seed, *self.neigh[seed][:k - 1]}
        elif arm == "route ruin":
            seed = rng.randint(1, n)
            where = {c: ri for ri, r in enumerate(routes) for c in r}
            targets = {where[seed]}
            for v in self.neigh[seed][:8]:
                if len(targets) >= 2:
                    break
                targets.add(where[v])
            for ri in targets:
                removed |= set(routes[ri])
        kept = [[c for c in r if c not in removed] for r in routes]
        return [r for r in kept if r], list(removed)

    def recreate(self, routes, removed):
        """Regret-2 insertion (penalised load), hand-written."""
        D, q, Q = self.D, self.q, self.Q
        pen = self.penalty
        routes = [list(r) for r in routes]
        loads = [sum(q[c] for c in r) for r in routes]
        todo = list(removed)
        self.rng.shuffle(todo)
        while todo:
            best_pick = None
            for c in todo:
                opts = []
                for ri, r in enumerate(routes):
                    over = max(0.0, loads[ri] + q[c] - Q) - max(0.0, loads[ri] - Q)
                    prev, bc, bp = 0, None, 0
                    for p, nx in enumerate(r + [0]):
                        inc = D[prev][c] + D[c][nx] - D[prev][nx]
                        if bc is None or inc < bc:
                            bc, bp = inc, p
                        prev = nx
                    opts.append((bc + pen * over, ri, bp))
                if len(routes) < self.K:
                    opts.append((D[0][c] + D[c][0], -1, 0))
                opts.sort(key=lambda t: t[0])
                regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else 1e18
                if best_pick is None or regret > best_pick[0]:
                    best_pick = (regret, c, opts[0])
            _, c, (_, ri, p) = best_pick
            if ri == -1:
                routes.append([c])
                loads.append(q[c])
            else:
                routes[ri].insert(p, c)
                loads[ri] += q[c]
            todo.remove(c)
        return routes

    # ----------------------------------------------------------- education
    def educate(self, routes, penalty=None):
        pen = self.penalty if penalty is None else penalty
        if self.fast:
            return self.ls.run(routes, pen, self.cfg.use_swap_star)
        return self.ls.run(routes, pen)

    def polish(self, ind):
        """Held-Karp exact order for each route; returns improved Indiv (or same)."""
        if not self.cfg.use_exact:
            return ind, 0
        routes, proven = self.exact.optimise_solution(ind.routes)
        new = self.make(routes)
        return (new if new.cost < ind.cost - 1e-9 else ind), proven

    # ----------------------------------------------------------------- run
    def run(self) -> Result2:
        cfg, tr = self.cfg, self.tr
        t0 = time.time()
        feas = SubPop(cfg, self.n)
        infeas = SubPop(cfg, self.n)
        best: Indiv | None = None
        history, first_feasible = [], None
        exact_gain = 0.0

        def add(ind):
            (feas if ind.feasible else infeas).add(ind)
            # route pool: only routes of near-best feasible solutions (<= 1% above best)
            if ind.feasible and (best is None or ind.cost <= best.cost * 1.01):
                self.pool.add_solution(ind.routes)

        def consider(ind, it, source):
            nonlocal best, first_feasible, exact_gain
            if not ind.feasible or (best is not None and ind.cost >= best.cost - 1e-9):
                return False
            before = ind.cost
            ind2, proven = self.polish(ind)
            if ind2 is not ind:
                exact_gain += before - ind2.cost
                add(ind2)
            best = ind2
            if first_feasible is None:
                first_feasible = before
            history.append((it, round(time.time() - t0, 3), best.cost))
            msg = (f"  iter {it:6d}  t={time.time() - t0:6.1f}s  best={best.cost:.2f}  "
                   f"routes={len(best.routes)}  via {source}"
                   + (f"  (Held-Karp exact: {before:.2f} -> {ind2.cost:.2f})" if ind2 is not ind else ""))
            tr.log(msg, force=True)
            if cfg.verbose:
                print(msg, flush=True)
            return True

        def educate_and_add(routes, it, source):
            out = self.educate(routes)
            ind = self.make(out)
            self.feas_log.append(ind.feasible)
            add(ind)
            imp = consider(ind, it, source)
            if not ind.feasible and self.rng.random() < cfg.repair_prob:
                rep = self.make(self.educate(out, self.penalty * 10))
                if rep.feasible:
                    add(rep)
                    imp = consider(rep, it, source + " + repair") or imp
            return ind, imp

        # ---------------------------------------------- 1. construction
        tr.section("PHASE 1 - HAND-WRITTEN CONSTRUCTION HEURISTICS (seed the population)")
        seeds = construction_portfolio(self.inst, tr if tr.enabled else None)
        tr.section("PHASE 1b - each construction improved by local search")
        for name, routes in seeds:
            raw = sum(self.route_cost(r) for r in routes)
            ind, _ = educate_and_add(self.split([c for r in routes for c in r]), 0, name)
            tr.log(f"   {name:24s} raw {raw:10.2f} ({len(routes):2d} routes) -> after LS "
                   f"{ind.cost:10.2f}  {'feasible' if ind.feasible else 'infeasible'}", force=True)
        customers = list(range(1, self.n + 1))
        while len(feas.inds) + len(infeas.inds) < 4 * cfg.mu:
            self.rng.shuffle(customers)
            educate_and_add(self.split(customers[:]), 0, "random tour")
            if time.time() - t0 > cfg.time_limit * 0.3:
                break

        # ---------------------------------------------- 2-4. evolution
        tr.section("PHASE 2 - HYBRID GENETIC SEARCH (every new best solution)")
        it, last_imp, last_restart = 0, 0, 0
        sp_reserve = cfg.sp_time if cfg.use_sp else 0.0
        while time.time() - t0 < cfg.time_limit - sp_reserve and it - last_imp < cfg.max_no_improve:
            it += 1
            feas.update_biased()
            infeas.update_biased()
            pop = feas.inds + infeas.inds
            p1 = self._tournament(pop)
            p2 = self._tournament(pop)
            child = self._ox(p1.tour, p2.tour)
            routes = self.split(child)
            # without the bandit the four operators are chosen uniformly at random (ablation)
            arm = self.bandit.select() if cfg.use_bandit else self.rng.choice(self.bandit.arms)
            if arm != "crossover only":
                kept, removed = self.ruin(routes, arm)
                routes = self.recreate(kept, removed)
            parent_best = min(p1.pcost, p2.pcost)
            ind, improved = educate_and_add(routes, it, arm)
            if cfg.use_bandit:
                reward = 1.0 if improved else (0.3 if ind.feasible and ind.cost < parent_best else 0.0)
                self.bandit.update(arm, reward)
            if improved:
                last_imp = it
            if it % 100 == 0:
                self._manage_penalty(infeas)
            if it - max(last_imp, last_restart) > cfg.restart_after:
                tr.log(f"  iter {it:6d}  restart (no improvement for {cfg.restart_after} iterations)",
                       force=True)
                feas, infeas = SubPop(cfg, self.n), SubPop(cfg, self.n)
                if best is not None:
                    feas.add(best)
                for _ in range(2 * cfg.mu):
                    self.rng.shuffle(customers)
                    educate_and_add(self.split(customers[:]), it, "restart")
                last_restart = it

        # ---------------------------------------------- 5. exact recombination
        sp_info = None
        if cfg.use_sp and best is not None:
            tr.section("PHASE 3 - EXACT SET PARTITIONING OVER THE ROUTE POOL")
            # add exact-order versions of pool routes (cheap for short routes)
            remaining = max(1.0, cfg.time_limit - (time.time() - t0))
            sp_routes, sp_info = solve_set_partitioning(self.pool, best.cost, remaining, tr)
            tr.log(f"   pool size {sp_info.get('pool_routes')}, B&B nodes {sp_info.get('nodes')}, "
                   f"finished={sp_info.get('proven_optimal_over_pool')}", force=True)
            if sp_routes is not None:
                cand = self.make(sp_routes)
                if cand.feasible and len(cand.routes) <= self.K and cand.cost < best.cost - 1e-9:
                    consider(cand, it, "set partitioning")

        runtime = time.time() - t0
        stats = {
            "engine": "HGS2 (v2.0)",
            "fast_O1_local_search": self.fast,
            "iterations": it,
            "final_penalty": round(self.penalty, 4),
            "local_search_moves_applied": dict(getattr(self.ls, "move_counts", {})),
            "bandit": self.bandit.table() if cfg.use_bandit else None,
            "held_karp_gain_total": round(exact_gain, 4),
            "held_karp_route_sets_solved": self.exact.calls,
            "route_pool_size": len(self.pool),
            "set_partitioning": sp_info,
            "construction_results": None,
        }
        if best is None:
            b = min(feas.inds + infeas.inds, key=lambda i: i.excess)
            return Result2(b.routes, b.cost, False, it, runtime, history, None, stats)
        return Result2(best.routes, best.cost, True, it, runtime, history, first_feasible, stats)

    # ----------------------------------------------------------- operators
    def _tournament(self, pop):
        a, b = self.rng.choice(pop), self.rng.choice(pop)
        return a if a.biased <= b.biased else b

    def _ox(self, p1, p2):
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

    def _manage_penalty(self, infeas):
        if not self.feas_log:
            return
        share = sum(self.feas_log[-100:]) / len(self.feas_log[-100:])
        if share < self.cfg.target_feasible - 0.05:
            self.penalty = min(1e5, self.penalty * 1.2)
        elif share > self.cfg.target_feasible + 0.05:
            self.penalty = max(0.1, self.penalty * 0.85)
        self.feas_log = []
        for i in infeas.inds:
            i.pcost = i.cost + self.penalty * i.excess
