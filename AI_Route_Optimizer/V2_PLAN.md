# AI-Based Green Route Optimizer: Version 2.0 Plan (manual-first)

**Status (29 Sep 2026):** Phases 1–5 below are implemented and tested. Every result in this file came from a single run of `python run_all.py`, and the full console log is in `results/run_log.txt`.

## 1. The principle: hand-written methods, proven where possible

The first v2.0 plan leaned on outside libraries: NumPy, Numba, C++ cores, a pre-trained neural model, and PyVRP/OR-Tools as building blocks. This version works the other way:

1. **Every algorithm in the solver is written by hand** using only the Python standard library. Each one follows a step list you can check by hand (see `MANUAL_METHODS.md`). The solver does not import PyVRP, OR-Tools, Numba, or any neural-network library.
2. **Exact parts come before heuristics wherever they fit.**
   - Held-Karp dynamic programming proves that the order within each route is optimal.
   - A hand-written set-partitioning branch & bound picks the best combination of routes from all the good routes found during the search.
3. **Every step leaves evidence.** `--trace` writes each savings merge, each construction result, each new best solution and the operator that found it, and the set-partitioning log.
4. **Every formula has a test.**
   - `tests/test_moves.py` checks the O(1) move deltas against full recomputation.
   - `tests/test_exact.py` checks Held-Karp against brute force.
   - `tests/test_set_partition.py` checks the branch & bound against brute force.

   The set-partitioning test caught a real bug: the depot bit was missing from the "all covered" mask, so the step had never been able to finish a cover. It was fixed before the final run.

## 2. Reference repositories: where they fall short and what we do instead

The manual-implementation repositories come from `CVRP-Manual-Implementation-Reference.pdf`. The state-of-the-art ones are from the GitHub review in the first plan.

| Repository | Method | Where it falls short | What v2.0 does instead |
|---|---|---|---|
| [yorak/VeRyPy](https://github.com/yorak/VeRyPy) | 15+ classical heuristics | Each heuristic runs on its own, with no shared constraint checker and no improvement phase | 17 constructions in one portfolio (`construction.py`). Each result is improved by the same local search and checked by one evaluator |
| [kavetinaveen/CWVRP](https://github.com/kavetinaveen/CWVRP) | Clarke-Wright, parallel and sequential | The sequential version can leave customers unserved | Our sequential CWS always serves every customer. We also run 8 Paessens (λ, μ) variants |
| [adamiaonr/decision-support-cvrp](https://github.com/adamiaonr/decision-support-cvrp) | CWS → SA → GA | Its CWS reports 865 on A-n32-k5. Parameters are fixed (SA cooling 0.005, GA population 120) | Our CWS gives 842; v2.0 gives 784 (optimal). The penalty adapts on its own, and the bandit picks operators while the solver runs |
| [vss2sn/cvrp](https://github.com/vss2sn/cvrp) | Greedy, LS, tabu, GA, SA | The GA's MakeValid step can still output an infeasible plan. Moves are random or first-found | Only plans with zero capacity excess are reported. Move choice is guided by granular neighbourhoods |
| [ivanchoff/TS_CVRP](https://github.com/ivanchoff/TS_CVRP), [AHalic/tabu-search](https://github.com/AHalic/tabu-search) | Sweep + tabu | Stops at a local optimum, and each move costs O(route) to evaluate | Each move is evaluated in O(1) from a hand-derived formula. Route orders are proven optimal by Held-Karp |
| [MekhyW](https://github.com/MekhyW) | Brute force + 2-opt with OpenMP/MPI | Brute force only works on tiny cases, and 2-opt alone is a local optimum | Held-Karp is an exact dynamic programme instead of brute force (m = 12 customers in 0.03 s instead of 12! permutations) |
| [msommacal/cvrp](https://github.com/msommacal/cvrp) | Route-first cluster-second | The fleet size is not limited: our re-implementation uses 6 routes on P-n40-k5, where only 5 are allowed | The Split decoder allows at most K routes |
| [manu-p-1/cvrp](https://github.com/manu-p-1/cvrp) | GA with BRX/CX/ERX/OX, population 600 | No local search, and results are "within 5% of optimal" | The GA is combined with local search on every child, and diversity is controlled |
| [zuzhaoye](https://github.com/zuzhaoye) CW notebook | Step-by-step CW printout | Covers only one heuristic | `--trace` records every phase of the solver |
| [vidalt/HGS-CVRP](https://github.com/vidalt/HGS-CVRP), [PyVRP](https://github.com/PyVRP/PyVRP) | State-of-the-art HGS in C++ | These are libraries we could not submit as our own | We wrote the HGS design ourselves from the papers, and added two exact steps that HGS does not have: Held-Karp route orders and set partitioning |
| [N-Wouda/ALNS](https://github.com/N-Wouda/ALNS), [ahottung/NLNS](https://github.com/ahottung/NLNS) | Bandit operator selection; destroy types | The ALNS approach is a library we could not submit; NLNS needs neural training | A hand-written UCB1 bandit learns during the run which of four ruin operators to use (random, proximity, route, or crossover only). It needs no training data |

## 3. What was built

| Phase | File | Content | Test |
|---|---|---|---|
| 1. Construction | `construction.py` | CW parallel + 8 Paessens variants, CW sequential, sweep from 6 start angles, nearest neighbour, cheapest insertion, all traced step by step | Every output goes through the evaluator |
| 2. Fast local search | `local_search2.py` | Relocate ×2, swap, or-opt ×2, 2-opt, 2-opt\* ×2, in-route moves, new-route move, SWAP\*. All evaluated in O(1). Granular 20, plus "last tested" memory | `tests/test_moves.py` |
| 3. HGS engine | `hgs2.py` | OX crossover, Split with at most K routes, feasible and infeasible subpopulations, biased fitness with broken-pairs diversity, adaptive penalty, restarts | End-to-end runs + `validate.py` |
| 4. Learning | `hgs2.py` (UCB1) | Bandit over 4 arms, regret-2 repair. The bandit's statistics are saved in `result.json` | Ablation (`--no-bandit`) |
| 5a. Exact route order | `exact_route.py` | Held-Karp DP. It is also exact for load-dependent CO₂. A certificate is printed in every report | `tests/test_exact.py` |
| 5b. Exact recombination | `set_partition.py` | Route pool + anytime branch & bound with a demand-share bound and a fleet bound | `tests/test_set_partition.py` |
| Comparison | `classic_methods.py` | Our re-implementations of the 9 reference-repo methods, all using the same parser and time limit | — |
| Evidence | `solver.py --trace`, `run_all.py` | `trace.txt`, engine statistics, 7-part experiment log | — |

## 4. Results of v2.0 (from `python run_all.py`, 2-core laptop, Python 3.14)

### 4.1 Official benchmarks: optimal, validated, and proven at the route level

| Benchmark | v2.0 | Optimum | Gap | Exact route check | Set partitioning over pool | Seeds reaching optimum (10 s each) | Mean time to best |
|---|---:|---:|---:|---|---|---:|---:|
| A-n32-k5 | 784 | 784 | 0% | 5/5 routes proven optimal | finished (12 routes) | 10/10 | 0.04 s |
| B-n31-k5 | 672 | 672 | 0% | 5/5 | finished (66 routes) | 10/10 | 0.01 s |
| P-n40-k5 | 458 | 458 | 0% | 5/5 | finished (25 routes) | 10/10 | 0.07 s |

All three were checked by `validate.py` ("VALID SOLUTION: YES").

### 4.2 Reference-repo methods compared with v2.0 (same parser; 10 s for the timed methods)

| Method (our re-implementation) | A-n32-k5 | B-n31-k5 | P-n40-k5 |
|---|---:|---:|---:|
| CWS parallel (VeRyPy / CWVRP / adamiaonr) | 842 (+7.40%) | 677 (+0.74%) | 516 (+12.66%) |
| CWS sequential (VeRyPy / CWVRP) | 904 (+15.31%) | 707 (+5.21%) | 512 (+11.79%) |
| Sweep (ivanchoff / VeRyPy) | 1561 (+99.11%) | 940 (+39.88%) | 671 (+46.51%) |
| Greedy nearest neighbour (vss2sn) | 1145 (+46.05%) | 887 (+31.99%) | 682 (+48.91%) |
| Route-first cluster-second (msommacal) | 887 (+13.14%) | 688 (+2.38%) | 546, **6 routes, infeasible** |
| CWS + 2-opt (MekhyW) | 829 (+5.74%) | 677 (+0.74%) | 516 (+12.66%) |
| CWS + simulated annealing (vss2sn / adamiaonr) | 827 (+5.48%) | 675 (+0.45%) | 507 (+10.70%) |
| Tabu search (vss2sn / ivanchoff / AHalic) | 785 (+0.13%) | 673 (+0.15%) | 458 (0%) |
| Simple GA, no local search (manu-p-1) | 870 (+10.97%) | 678 (+0.89%) | 500 (+9.17%) |
| **v2.0 (ours)** | **784 (0%)** | **672 (0%)** | **458 (0%)** |

Tabu search is the strongest of the classic methods. Even so, it misses the optimum by 1 unit on both A and B.

### 4.3 Generalisation: 8 CVRPLIB instances not used during development (20 s, seed 1)

| Instance | A-n33-k5 | A-n45-k6 | A-n60-k9 | B-n45-k5 | B-n57-k9 | P-n50-k7 | P-n76-k5 | E-n51-k5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.0 (30 s) | 661 | 966 | 1354 | 751 | 1601 | 554 | 627 | 521 |
| **v2.0 (20 s)** | **661** | **944** | **1354** | **751** | **1598** | **554** | **627** | **521** |
| Optimum | 661 | 944 | 1354 | 751 | 1598 | 554 | 627 | 521 |

**8/8 optimal** with v2.0, compared with 6/8 for v1.0. The two instances v1.0 missed, A-n45-k6 and B-n57-k9, are now solved.

### 4.4 Scaling and ablation on Uchoa X instances (45 s, seed 1; fleet = BKS routes + 2)

| Instance | BKS | v1.0 | v2.0 full | v2.0 without bandit | v2.0 without exact steps | v2.0 children vs v1.0 |
|---|---:|---:|---:|---:|---:|---:|
| X-n101-k25 | 27591 | 28014 (+1.53%) | **27657 (+0.24%)** | 27659 (+0.25%) | 27657 (+0.24%) | 2060 vs 278 (7.4×) |
| X-n157-k13 | 16876 | 17238 (+2.15%) | 16940 (+0.38%) | **16915 (+0.23%)** | 16938 (+0.37%) | 649 vs 105 (6.2×) |
| X-n200-k36 | 58578 | 60380 (+3.08%) | **59011 (+0.74%)** | 59282 (+1.20%) | 59036 (+0.78%) | 610 vs 73 (8.4×) |

With a 60 s limit (the same budget as the earlier stress test), v2.0 reached 27636 (+0.16%), 16927 (+0.30%) and 59011 (+0.74%). That is an **average gap of 0.40%**, down from 1.80% for v1.0, which meets the plan's target of 0.5% or less. For reference, PyVRP 0.14, which runs compiled C++, reached 0.00 / 0.00 / 0.30% on the same machine.

**Honest reading of the ablation.** Each cell above is a single-seed run.

- **O(1) moves and HGS diversity:** these two give most of the gain over v1.0.
- **Bandit:** it helps clearly on X-n200 (+0.74% compared with +1.20% without it) but not on X-n157, so its effect needs a 10-seed test before we claim it on a slide.
- **Exact steps:** removing them costs little. On instances this size, the solution quality comes from the heuristic search. The exact steps add proof on small instances: each route is certified optimal, and on A/B/P the set-partitioning search finished and proved no better mix exists in the pool. On X instances, the simple bound is too weak for set partitioning to finish within its 10 s budget.

### 4.5 Self-created case: SCG-CPAC Narathiwat (25 customers, 6 trucks × 12 t)

| Plan | km | Fuel (L) | CO₂ (kg) | Trucks |
|---|---:|---:|---:|---:|
| Sweep + fill trucks (manual-style plan) | 877.0 | 238.1 | 638.1 | 6 |
| Nearest neighbour | 828.7 | 211.5 | 566.9 | 6 |
| v2.0, minimise distance | **641.4** | 174.0 | 466.4 | **5** |
| v2.0, minimise CO₂ | 644.2 | **170.4** | **456.7** | 6 |

- Every route in both v2.0 plans has a Held-Karp certificate. For the CO₂ objective, this uses the load-dependent DP.
- The CO₂ plan cuts emissions by **28.4%** compared with the manual-style plan.

## 5. Next steps (manual-first)

| Priority | Task | How we keep it manual and precise |
|---|---|---|
| 1 | 10-seed ablation on X instances (bandit, SWAP\*, exact steps) | Same `run_all.py` code, just more seeds. Report mean ± standard deviation |
| 2 | A stronger set-partitioning bound | Write a small simplex method by hand to get the LP relaxation bound, so set partitioning can finish on 100+ customers |
| 3 | Held-Karp up to 16 customers on the final solution only | Already takes 0.9 s at m = 16; the certificate would then cover longer routes |
| 4 | Real road distances for Narathiwat | Build the matrix once, save it in the JSON, and have the solver read only that matrix. OSRM is used to build the data, not inside the solver |
| 5 | Time windows and an EV truck | Add a time-window penalty in the same O(1) style, with a test for every new formula |
| 6 | Presentation | `trace.txt` excerpts, the comparison table (4.2), the scaling table (4.4), and route certificates |

## 6. Rules check

| Rule | How v2.0 meets it |
|---|---|
| The team's own solver | Every algorithm is hand-written in `construction.py`, `local_search2.py`, `hgs2.py`, `exact_route.py` and `set_partition.py`, using only the standard library |
| No hard-coded solutions | All data comes from the input file. The optimum is read after solving, only to print the gap |
| Evidence not edited by hand | `validate.py` recomputes everything independently. `result.json` stores the input SHA-256, the exact command, the seed, and the engine statistics. `run_log.txt` and `trace.txt` are written by the program itself |
