# Hand-written methods in v2.0: step-by-step reference

Every algorithm in the v2.0 solver was written by hand using only the Python standard library. The solver does not import PyVRP, OR-Tools, Numba or any neural-network library. Each method below lists:

- its steps, as they appear in the code
- the file that contains it
- how we test that it is correct
- the paper it comes from

Run `python solver.py <file> --trace` to write a `trace.txt` that records each step as it happens. It shows every savings merge, every construction result, every new best solution and which operator found it, and the set-partitioning log.

---

## 1. Construction heuristics (`construction.py`)

These build the first population. Each one is a different, repeatable way to plan routes by hand.

### 1.1 Clarke-Wright savings, parallel version (Clarke & Wright 1964; Paessens 1988)
1. Start with one route per customer, 0 → i → 0.
2. For every pair (i, j), compute the saving *s(i,j) = d(0,i) + d(0,j) − λ·d(i,j) + μ·|d(0,i) − d(0,j)|*.
3. Sort the savings from largest to smallest.
4. Go down the list. Merge the route ending in i with the route starting in j when all of these hold:
   - i and j are at the ends of different routes
   - the merged load is at most Q
   - the saving is positive
5. We run the classic version (λ = 1, μ = 0) plus 8 Paessens variants, λ ∈ {0.6, 0.8, 1.2, 1.4} × μ ∈ {0, 0.5}.

The trace prints the top-10 savings and every merge, like the zuzhaoye notebook does.

### 1.2 Clarke-Wright savings, sequential version
Grow one route at a time from the best saving that touches either end of the route. When no more merges fit, open a new route. **Fix:** after the last route, any customer that was never merged is kept as its own route. The sequential code in kavetinaveen/CWVRP can leave these customers unserved.

### 1.3 Sweep (Gillett & Miller 1974)
1. Compute each customer's polar angle around the depot.
2. Sort by angle, starting from 6 different start angles (0°, 60°, …, 300°).
3. Fill trucks in that order and start a new truck when the next customer doesn't fit.

### 1.4 Nearest neighbour
Drive to the closest unvisited customer that still fits in the truck. Return to the depot when none fits.

### 1.5 Cheapest insertion
Add customers to routes one at a time, each where it adds the least distance.

### 1.6 Route-first, cluster-second: the Split decoder (Beasley 1983; Prins 2004)
1. Treat the customers as one giant tour (a permutation).
2. Build a graph with nodes 0..n. The arc (i, j) is the route tour[i+1..j], used only if its load is at most Q.
3. Find the shortest path from 0 to n. It gives the best way to cut the tour into routes.
4. **Fleet-limited version:** the same dynamic programme with a route-count index keeps at most K routes. The CWS and sweep repos do not check the fleet limit.

---

## 2. Local search with O(1) move evaluation (`local_search2.py`)

Notation: u is in route A at position i, with pu = pred(u) and x = succ(u). v is in route B at position j, with pv = pred(v) and y = succ(v). The depot is 0.

| Move | Change in cost (written out by hand) |
|---|---|
| Relocate u after v | d(pu,x) − d(pu,u) − d(u,x) + d(v,u) + d(u,y) − d(v,y) |
| Relocate u before v | d(pu,x) − d(pu,u) − d(u,x) + d(pv,u) + d(u,v) − d(pv,v) |
| Swap u ↔ v | d(pu,v) + d(v,x) − d(pu,u) − d(u,x) + d(pv,u) + d(u,y) − d(pv,v) − d(v,y) |
| Or-opt (u,x) after v | d(pu,xx) − d(pu,u) − d(x,xx) + d(v,u) + d(x,y) − d(v,y) (and the reversed pair) |
| 2-opt (same route) | d(u,v) + d(x,y) − d(u,x) − d(v,y) |
| 2-opt\*, tails | d(u,y) + d(v,x) − d(u,x) − d(v,y) |
| 2-opt\*, reversed | d(u,v) + d(x,y) − d(u,x) − d(v,y) |
| SWAP\* (Vidal 2022) | exchange u and v between two routes, each re-inserted at its best position in the other route, using the 3 cheapest pre-computed insertion points |

The capacity term is *P · max(0, load − Q)* for each route, computed from cached route loads and prefix loads.

Speed-ups:
- **Granular neighbourhoods:** each u is tested only against its 20 nearest customers (Toth & Vigo 2003).
- **"Last tested" memory:** skip a pair (u, v) when neither route has changed since u was last tested. The same rule applies to SWAP\* route pairs.

Measured on X-n101-k25, one full descent from a random start takes **0.012 s**. The same descent took about 0.1 s before the last-tested memory was added.

**Correctness:** `tests/test_moves.py` checks every formula against a full recomputation on 40 random instances and on the benchmarks. With `debug_check=True`, the code asserts that every applied move really lowers the cost.

---

## 3. Hybrid genetic search (`hgs2.py`), after Vidal et al. (2012) and Vidal (2022)

1. **Chromosome:** a giant tour of the customers, turned into routes by Split.
2. **Parent selection:** binary tournament on biased fitness, which combines cost rank and diversity rank. Diversity is measured by the broken-pairs distance to the 4 closest individuals.
3. **Crossover:** Order Crossover (OX).
4. **Mutation:** chosen by the UCB1 bandit, see section 4.
5. **Education:** the local search from section 2, with a capacity penalty. If the child is still over capacity, it is repaired half the time with a 10× penalty.
6. **Two sub-populations:** one for feasible and one for infeasible solutions. Each holds between μ = 25 and μ + λ = 65 individuals. When full, the worst by biased fitness are removed, clones first.
7. **Adaptive penalty:** every 100 children, the penalty is multiplied by 1.2 if fewer than 20% are feasible, and by 0.85 if more are.
8. **Restart:** after 4000 children with no improvement, the population is rebuilt around the best solution.

---

## 4. Adaptive operator choice: UCB1 bandit (Auer et al. 2002)

The bandit chooses between 4 arms:
- crossover only
- random ruin
- proximity ruin (a customer and its nearest neighbours)
- route ruin (two nearby routes)

Every ruin arm is followed by a **regret-2 repair**: the customer whose second-best insertion is most expensive compared with its best is inserted first.

- **Reward:** 1 for a new best solution, 0.3 for a feasible child better than both parents, otherwise 0.
- **Selection rule:** pick the arm with the highest *mean + 0.5·√(ln t / n_arm)*.
- **Learning:** the bandit learns during each run and needs no training data. Each arm's usage count and mean reward are saved in `result.json → engine_statistics.bandit`.
- **Ablation:** `--no-bandit` picks the arm uniformly at random instead.

---

## 5. Exact route order: Held-Karp dynamic programming (Held & Karp 1962)

For a route with m ≤ 12 customers:

- *C(S, j)* = cheapest path that starts at the depot, visits every customer in set S once, and ends at j
- *C({j}, j) = d(0, j)*
- *C(S, j) = min over i ∈ S∖{j} of C(S∖{j}, i) + d(i, j)*
- optimum = *min over j of C(all, j) + d(j, 0)*

For the CO₂ objective, the arc cost depends on the truck's load. The load on an arc is known from S: it is the total demand minus what has already been delivered. So the same recursion is still exact.

- **Timing:** m = 12 takes 0.03 s and m = 16 takes 0.9 s.
- **Caching:** results are cached per customer set.
- **When it runs:** on every new best solution.
- **Certificate:** after solving, the report prints "Exact route check: k/K routes have a proven-optimal order". A local-search optimum (2-opt, tabu or SA) cannot give this guarantee.
- **Correctness:** `tests/test_exact.py` matches brute force over all permutations, for both distance and CO₂.

---

## 6. Exact recombination: set partitioning (`set_partition.py`)

- **Route pool:** all routes from feasible solutions within 1% of the best. For each customer set, only the cheapest order is kept.
- **Problem:** choose routes from the pool so that every customer is covered exactly once, using at most K routes, at the lowest total cost.
- **Method:** a hand-written depth-first branch & bound.
  - **Branching:** pick the uncovered customer with the fewest compatible routes, and try its routes cheapest share first.
  - **Bound 1:** cost so far + Σ over uncovered customers c of the minimum of *cost(r)·q_c / load(r)* over routes r. This is a valid lower bound because every completion splits its cost in exactly this way.
  - **Bound 2:** routes used + ⌈remaining demand / Q⌉ ≤ K.
  - **Anytime:** it starts with the GA's best cost as the upper bound, stops at its time budget (15% of the run, at most 10 s), and keeps the best exact cover found.
- **Correctness:** `tests/test_set_partition.py` matches brute-force enumeration of all exact covers on 15 random pools.
- **Honest limitation:** on the 30–40 customer benchmarks, the search finishes and proves optimality over the pool. On 100+ customers the simple bound is too weak to finish in 10 s. A future version would need an LP-based bound, i.e. a hand-written simplex method.

---

## 7. Reference-repo methods re-implemented for comparison (`classic_methods.py`)

| Method | Reference repositories (CVRP-Manual-Implementation-Reference.pdf) |
|---|---|
| CWS parallel | yorak/VeRyPy (ps), kavetinaveen/CWVRP, adamiaonr/decision-support-cvrp |
| CWS sequential | yorak/VeRyPy (ss), kavetinaveen/CWVRP |
| Sweep | ivanchoff/TS_CVRP (phase 1), yorak/VeRyPy (swp) |
| Greedy nearest neighbour | vss2sn/cvrp Algorithm 1 |
| Route-first cluster-second | msommacal/cvrp |
| CWS + 2-opt | MekhyW (2-opt part) |
| CWS + simulated annealing | vss2sn/cvrp Algorithm 6, adamiaonr (CWS → SA) |
| Tabu search (relocate, aspiration) | vss2sn/cvrp, ivanchoff/TS_CVRP, AHalic/tabu-search |
| Simple GA without local search (OX, 0.15 mutation) | manu-p-1/cvrp |

We wrote this code ourselves from the published step lists; it is not the original code. That way every method uses the same parser, distance matrix and time limit, so the comparison is fair.

---

## References

- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47, 235–256. https://doi.org/10.1023/A:1013689704352
- Beasley, J. E. (1983). Route first—Cluster second methods for vehicle routing. *Omega*, 11(4), 403–408. https://doi.org/10.1016/0305-0483(83)90033-6
- Clarke, G., & Wright, J. W. (1964). Scheduling of vehicles from a central depot to a number of delivery points. *Operations Research*, 12(4), 568–581. https://doi.org/10.1287/opre.12.4.568
- Gillett, B. E., & Miller, L. R. (1974). A heuristic algorithm for the vehicle-dispatch problem. *Operations Research*, 22(2), 340–349. https://doi.org/10.1287/opre.22.2.340
- Held, M., & Karp, R. M. (1962). A dynamic programming approach to sequencing problems. *J. SIAM*, 10(1), 196–210. https://doi.org/10.1137/0110015
- Paessens, H. (1988). The savings algorithm for the vehicle routing problem. *European Journal of Operational Research*, 34(3), 336–344. https://doi.org/10.1016/0377-2217(88)90154-3
- Prins, C. (2004). A simple and effective evolutionary algorithm for the vehicle routing problem. *Computers & Operations Research*, 31(12), 1985–2002. https://doi.org/10.1016/S0305-0548(03)00158-8
- Toth, P., & Vigo, D. (2003). The granular tabu search and its application to the vehicle-routing problem. *INFORMS Journal on Computing*, 15(4), 333–346. https://doi.org/10.1287/ijoc.15.4.333.24890
- Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., & Rei, W. (2012). A hybrid genetic algorithm for multidepot and periodic vehicle routing problems. *Operations Research*, 60(3), 611–624. https://doi.org/10.1287/opre.1120.1048
- Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood. *Computers & Operations Research*, 140, 105643. https://doi.org/10.1016/j.cor.2021.105643
