# AI-Based Green Route Optimizer v2.0

A generalized CVRP solver built for **AI Hackathon: Route Optimization 2026, "AI Engineering for Green Logistics"**.

The same solver and settings are used for every CVRP instance. The program reads the benchmark file and builds the distance matrix itself, then works out the routes. It does not contain any stored answers or code written for a specific instance.

**Changes in v2.0.** Every algorithm is written by hand in pure Python, using only the standard library. The solver imports no PyVRP, OR-Tools, Numba or neural-network library. Two exact steps were added:

- **Held-Karp dynamic programming** proves that the stop order within each route is optimal.
- **Set-partitioning branch & bound** recombines the best routes found during the search.

`--trace` writes every solving step to a file. See `MANUAL_METHODS.md` for how each method works step by step, and `V2_PLAN.md` for the plan and the full results.

## Quick start

Requires Python 3.9+. `matplotlib` is needed only for the figures.

```bash
pip install matplotlib

# solve the three official benchmarks with one command and the same settings
python solver.py benchmarks/A-n32-k5.vrp benchmarks/B-n31-k5.vrp benchmarks/P-n40-k5.vrp --time 30 --seed 1 --trace

# check a solution independently (this script imports no solver code)
python validate.py benchmarks/A-n32-k5.vrp results/A-n32-k5/solution.sol

# self-created case: minimise distance, then minimise CO2
python solver.py own_case/scg_cpac_narathiwat.json
python solver.py own_case/scg_cpac_narathiwat.json --objective co2

# correctness tests for the hand-written formulas
python tests/test_moves.py && python tests/test_exact.py && python tests/test_set_partition.py

# reproduce every number in this README (about 20 min; --fast takes about 5 min)
python run_all.py
```

Options:

- `--time`: time limit per instance, in seconds
- `--seed`: random seed
- `--vehicles`: override the fleet size
- `--objective distance|co2`
- `--out`: output folder
- `--quiet`
- `--trace`: write `trace.txt`
- `--engine v1|v2`
- ablation switches: `--no-bandit`, `--no-swap-star`, `--no-exact`, `--no-sp`

## Results: official benchmarks

All three were solved in one run of `solver.py` with the same settings (`--time 30 --seed 1`).

| Benchmark | Customers | Vehicles | Capacity | Our distance | Published optimum | Gap | Feasible | Exact route check | Seeds reaching optimum |
|---|---:|---:|---:|---:|---:|---:|:---:|---|:---:|
| A-n32-k5 | 31 | 5 | 100 | **784** | 784 | 0.00% | YES | 5/5 routes proven optimal | 10/10 |
| B-n31-k5 | 30 | 5 | 100 | **672** | 672 | 0.00% | YES | 5/5 | 10/10 |
| P-n40-k5 | 39 | 5 | 140 | **458** | 458 | 0.00% | YES | 5/5 | 10/10 |

`validate.py` is a separate checker that uses no solver code. It re-checked every solution:

- every customer is visited exactly once
- no truck is over capacity
- no more than 5 routes are used
- the recomputed distance matches the reported one

The output is in `results/<benchmark>/validation.txt`, and the full solving steps are in `results/<benchmark>/trace.txt`.

### Compared with the methods used in the reference repositories

We wrote our own versions of these methods (`classic_methods.py`). They all use the same parser and distance matrix, and each timed method gets 10 s.

| Method | A-n32-k5 | B-n31-k5 | P-n40-k5 |
|---|---:|---:|---:|
| Clarke-Wright parallel (VeRyPy, CWVRP, adamiaonr) | 842 (+7.4%) | 677 (+0.7%) | 516 (+12.7%) |
| Sweep (ivanchoff, VeRyPy) | 1561 (+99.1%) | 940 (+39.9%) | 671 (+46.5%) |
| Greedy nearest neighbour (vss2sn) | 1145 (+46.1%) | 887 (+32.0%) | 682 (+48.9%) |
| Route-first cluster-second (msommacal) | 887 (+13.1%) | 688 (+2.4%) | 546, 6 routes: **infeasible** |
| CWS + simulated annealing (vss2sn, adamiaonr) | 827 (+5.5%) | 675 (+0.4%) | 507 (+10.7%) |
| Tabu search (vss2sn, ivanchoff, AHalic) | 785 (+0.1%) | 673 (+0.1%) | 458 (0%) |
| Simple GA without local search (manu-p-1) | 870 (+11.0%) | 678 (+0.9%) | 500 (+9.2%) |
| **v2.0 (ours)** | **784 (0%)** | **672 (0%)** | **458 (0%)** |

The full table, including sequential CWS and CWS + 2-opt, is in `results/comparison_reference_methods.csv`.

### Generalization: 8 instances not used during development (20 s each)

v2.0 found the optimum on all 8 instances. v1.0 found 6 of 8; it missed A-n45-k6 (944, v1 found 966) and B-n57-k9 (1598, v1 found 1601).

| A-n33-k5 | A-n45-k6 | A-n60-k9 | B-n45-k5 | B-n57-k9 | P-n50-k7 | P-n76-k5 | E-n51-k5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 661 | 944 | 1354 | 751 | 1598 | 554 | 627 | 521 |

### Scaling: larger Uchoa X instances (100–200 customers, 45 s each)

| Instance | Best known | v1.0 | v2.0 |
|---|---:|---:|---:|
| X-n101-k25 | 27591 | 28014 (+1.53%) | 27657 (+0.24%) |
| X-n157-k13 | 16876 | 17238 (+2.15%) | 16940 (+0.38%) |
| X-n200-k36 | 58578 | 60380 (+3.08%) | 59011 (+0.74%) |

With the O(1) move evaluation, v2.0 produces 6–8× more GA children per second than v1.0. The ablation results (bandit off, exact steps off) are in `V2_PLAN.md`.

## How the solver works

```
 .vrp / .json ──► cvrp_parser.py ──► distance matrix (TSPLIB EUC_2D rule)
                                          │
 1 CONSTRUCT  construction.py   CW savings (9 variants), sequential CW, sweep (6 angles),
                                nearest neighbour, cheapest insertion  -> each improved by LS
 2 EVOLVE     hgs2.py           giant tour + OX crossover + Split (<= K routes)
                                feasible / infeasible sub-populations, diversity-aware fitness,
                                adaptive capacity penalty, restarts
 3 IMPROVE    local_search2.py  relocate, swap, or-opt, 2-opt, 2-opt*, SWAP*  (O(1) deltas,
                                granular 20 nearest neighbours)
 4 LEARN      UCB1 bandit       picks the mutation: none / random / proximity / route ruin
                                + regret-2 repair, from the rewards seen in this run
 5 PROVE      exact_route.py    Held-Karp DP: each route's visiting order is optimal
 6 RECOMBINE  set_partition.py  exact branch & bound over the pool of good routes
                                          │
                                          ▼
          evaluation.py (distance, loads, feasibility, fuel, CO2)  ->  validate.py (independent)
          results/<name>/ solution.sol · result.json · result.txt · trace.txt · *.png
```

## Anti-hard-coding and evidence integrity

| Rule | How we meet it |
|---|---|
| No hard-coded answers | The solver takes all data from the input file. The published optimum is read from the file's COMMENT line by `read_reference_value()` after solving, only to print the gap. The optimizer never sees it. |
| One generalized solver | The same `solver.py` and settings solve every `.vrp`/`.json` file, including 8 extra CVRPLIB instances and 3 X instances with 100–200 customers. |
| Results not edited by hand | `validate.py` recomputes everything from the original benchmark file. `result.json` stores the SHA-256 of the input file, the exact command, seed, timestamp, Python version, and convergence history. |
| Reproducible | The random seed is fixed. `python run_all.py` regenerates all results. `results/run_log.txt` is the full console log, and `trace.txt` records every solving step. |

## Self-created case (Mission 3): SCG-CPAC building materials in Narathiwat

`own_case/scg_cpac_narathiwat.json` (created by `own_case/make_own_case.py`):

- **Depot**: a hypothetical CPAC regional distribution centre in Mueang Narathiwat
- **25 customers** (hardware stores, construction sites, precast yards, housing projects) in 12 districts: Mueang Narathiwat, Tak Bai, Bacho, Yi-ngo, Ra-ngae, Rueso, Si Sakhon, Waeng, Sukhirin, Su-ngai Kolok, Su-ngai Padi, Chanae
- **Demand**: 0.8–3.6 t of cement/concrete products per customer, 58.7 t in total
- **Fleet**: 6 six-wheel trucks with 12 t payload each
- **Distances**: district-office coordinates come from Wikipedia. Customers are placed 1–4 km around each office using a fixed seed. Straight-line km are multiplied by a road circuity factor of 1.3.
- **Green model**: fuel use depends on load. It goes from 0.22 L/km when empty to 0.33 L/km at full load (our planning assumption), with CO2 = 2.68 kg per litre of diesel ([NZ Ministry for the Environment emission factors](https://environment.govt.nz/assets/Publications/Files/voluntary-ghg-reporting-summary-tables-emissions-factors-2015.pdf); Thailand's [TGO emission factor table](https://thaicarbonlabel.tgo.or.th/index.php?lang=EN&mod=YjNKbllXNXBlbUYwYVc5dVgyVnRhWE56YVc5dQ) lists a similar value of about 2.70 for diesel).

| Plan | Distance (km) | Fuel (L) | CO2 (kg) | Trucks |
|---|---:|---:|---:|---:|
| Sweep + fill trucks (manual-style plan) | 877.0 | 238.1 | 638.1 | 6 |
| Nearest neighbour | 828.7 | 211.5 | 566.9 | 6 |
| v2.0 – minimise distance | **641.4** | 174.0 | 466.4 | **5** |
| v2.0 – minimise CO2 | 644.2 | **170.4** | **456.7** | 6 |

Compared with the manual-style plan, the distance-optimal plan cuts distance by 26.9% and CO2 by 26.9%. The CO2-optimal plan cuts CO2 by 28.4%. It saves another 9.7 kg of CO2 (−2.1%) compared with the distance-optimal plan, even though it drives 2.8 km more. Every route in both plans has a Held-Karp certificate, and the CO2 plan's certificate uses the load-dependent DP. It does this by dropping heavy loads early and using a sixth truck. This is a real trade-off for discussion: less fuel and CO2 in exchange for one more driver and truck.

## Project structure

```
AI_Route_Optimizer/
├── solver.py              main program (CLI, reports, outputs, exact-route certificate)
├── construction.py        hand-written construction heuristics + step-by-step Tracer
├── local_search2.py       v2 local search, O(1) move deltas, SWAP*
├── hgs2.py                v2 hybrid genetic search + UCB1 bandit + ruin & recreate
├── exact_route.py         Held-Karp dynamic programming (distance and load-dependent CO2)
├── set_partition.py       route pool + exact set-partitioning branch & bound
├── classic_methods.py     our re-implementations of the reference-repo methods (comparison)
├── genetic_algorithm.py   v1.0 engine (still available: --engine v1)
├── local_search.py        v1.0 local search (also used for the CO2 objective)
├── cvrp_parser.py         reads .vrp (TSPLIB/CVRPLIB) and .json instances
├── evaluation.py          objectives (distance / CO2) and full feasibility report
├── baselines.py           manual-style plans for the own case
├── validate.py            independent solution checker (no solver imports)
├── plots.py, render_evidence.py   route maps, convergence charts, console screenshots
├── run_all.py             reproduces every experiment
├── tests/                 test_moves.py, test_exact.py, test_set_partition.py
├── MANUAL_METHODS.md      step-by-step description of every hand-written method
├── V2_PLAN.md             manual-first v2.0 plan, where-others-lack table, full results
├── benchmarks/            A-n32-k5, B-n31-k5, P-n40-k5 (original files)
├── extra_tests/           8 CVRPLIB instances + x/ (X-n101-k25, X-n157-k13, X-n200-k36 with BKS .sol)
├── own_case/              self-created case (generator + JSON)
├── results/               outputs of `python run_all.py` (v2.0), run_log.txt
└── results_v1/            earlier v1.0 outputs, kept for comparison
```

Benchmark files are the original Augerat et al. instances from the [COIN-OR SYMPHONY VRP data set](https://www.coin-or.org/SYMPHONY/branchandcut/VRP/data/index.htm). The same instances are also published on CVRPLIB.

## Limitations and future work

- Distances are Euclidean. A real deployment would use a road-network distance/time matrix (e.g., OSRM or Google Maps).
- Time windows, driver working hours, and a mixed fleet are not modelled yet. The GA/Split design can be extended to include them.
- The fuel model is a simple linear load model. It could be calibrated with real truck telematics data.
- The solver is written in pure Python so that every step can be read and checked. On 200 customers it is within 0.74% of the best known solution in 45 s; compiled solvers such as PyVRP reach about 0.3% in the same time.
- On 100+ customers, set partitioning cannot finish within its 10 s budget because its bound is too simple. A hand-written LP bound is planned next.
