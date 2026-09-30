# CVRP Solver Research

A catalogue of open-source solvers for the Capacitated Vehicle Routing Problem (CVRP) on GitHub. It lists the main solving methods, from classic hand-worked heuristics to modern metaheuristics, with academic references.

## What is CVRP?

The **Capacitated Vehicle Routing Problem** asks for the cheapest set of routes for a fleet of identical vehicles. All vehicles start and end at one depot and together serve a set of customers, each with a known demand. Each customer is visited exactly once, and the total demand on a route cannot exceed the vehicle capacity *Q*. The objective is usually to minimise total travel distance, sometimes with a limit on the number of vehicles.

CVRP generalises the Travelling Salesman Problem, so it is NP-hard. Exact methods (branch-and-cut-and-price) can solve instances with a few hundred customers. Larger instances are normally tackled with heuristics and metaheuristics. Standard benchmark sets such as Augerat A/B/P, Christofides–Eilon E, and Uchoa X are collected on [CVRPLIB](http://vrp.galgos.inf.puc-rio.br/).

## Popular CVRP solver repositories

Star counts were taken from the GitHub API on 29 September 2026.

| Repository | Stars | Language | Algorithm | Link |
|---|---:|---|---|---|
| google/or-tools | 14,125 | C++ (Python/Java/C# wrappers) | Constraint programming routing library: construction heuristics + guided local search / other metaheuristics | [github.com/google/or-tools](https://github.com/google/or-tools) |
| VROOM-Project/vroom | 1,873 | C++ | Fast heuristic engine for real-world VRPs (CVRP, VRPTW, PDPTW) with road-network matrices (OSRM, ORS, Valhalla) | [github.com/VROOM-Project/vroom](https://github.com/VROOM-Project/vroom) |
| PyVRP/PyVRP | 703 | Python (C++ core) | Iterated local search, HGS-derived; many VRP variants | [github.com/PyVRP/PyVRP](https://github.com/PyVRP/PyVRP) |
| vidalt/HGS-CVRP | 443 | C++ | Hybrid Genetic Search with advanced diversity control + SWAP* neighbourhood | [github.com/vidalt/HGS-CVRP](https://github.com/vidalt/HGS-CVRP) |
| yorak/VeRyPy | 288 | Python | 15+ classical heuristics written from scratch (savings, sweep, insertion, petal, GAP, …) + 11 local-search operators | [github.com/yorak/VeRyPy](https://github.com/yorak/VeRyPy) |
| Kuifje02/vrpy (now romain-montagne/vrpy) | 202 | Python | Column generation (exact/heuristic pricing) for VRP variants | [github.com/romain-montagne/vrpy](https://github.com/romain-montagne/vrpy) |
| vss2sn/cvrp | 19 | C++ | Greedy, local search, tabu search, genetic algorithm, simulated annealing, hybrid — with step-by-step documentation | [github.com/vss2sn/cvrp](https://github.com/vss2sn/cvrp) |
| graphhopper/jsprit | 1,830 | Java | Ruin-and-recreate metaheuristic for rich VRPs | [github.com/graphhopper/jsprit](https://github.com/graphhopper/jsprit) |
| N-Wouda/ALNS | 658 | Python | Adaptive Large Neighbourhood Search framework (CVRP example) | [github.com/N-Wouda/ALNS](https://github.com/N-Wouda/ALNS) |
| acco93/filo2 | 33 | C++ | Fast iterated local search for very large CVRP instances | [github.com/acco93/filo2](https://github.com/acco93/filo2) |

`Kuifje02/vrpy` now redirects to `romain-montagne/vrpy`.

## Our solver: AI_Route_Optimizer v2.0

The [`AI_Route_Optimizer/`](AI_Route_Optimizer/) folder contains our own CVRP solver, written for AI Hackathon: Route Optimization 2026. Every algorithm is hand-written in standard-library Python: construction heuristics, O(1) local search with SWAP*, hybrid genetic search with a UCB1 bandit, Held-Karp exact route ordering, and set-partitioning branch & bound.

| Test | Result |
|---|---|
| A-n32-k5 / B-n31-k5 / P-n40-k5 | 784 / 672 / 458 (all optimal, 10/10 seeds) |
| 8 extra CVRPLIB instances | 8/8 optimal |
| X-n101 / X-n157 / X-n200 (60 s) | +0.16% / +0.30% / +0.74% from best known |

See [`AI_Route_Optimizer/README.md`](AI_Route_Optimizer/README.md) for usage, [`MANUAL_METHODS.md`](AI_Route_Optimizer/MANUAL_METHODS.md) for step-by-step method descriptions, and [`V2_PLAN.md`](AI_Route_Optimizer/V2_PLAN.md) for the plan and full results.

## Algorithm categories

### Constructive heuristics
- **Clarke-Wright Savings**: start with one route per customer, then merge routes in decreasing order of saving *s(i,j) = d(0,i) + d(0,j) − d(i,j)* whenever the merged route fits in the vehicle. Parallel and sequential versions exist.
- **Sweep**: sort customers by their polar angle around the depot, then fill vehicles in that order.
- **Nearest Neighbor**: repeatedly drive to the closest unvisited customer that still fits in the vehicle.
- **Insertion**: insert customers one at a time where they add the least cost (cheapest insertion) or where delaying them would cost the most (regret insertion).

### Local search
- **2-opt**: remove two edges within one route and reconnect it by reversing the segment between them.
- **3-opt**: remove three edges within one route and reconnect it in the best way.
- **2-opt\***: swap the tails of two different routes.
- **3-opt\***: the inter-route version of 3-opt.
- **Relocate**: move a customer (or a chain of customers, called or-opt) to another position or route.
- **Swap**: exchange two customers, in the same route or in different routes. SWAP\* (Vidal 2022) re-inserts each customer at its best position in the other route.

### Metaheuristics
- **Tabu Search**: always take the best allowed move, even if it makes the solution worse. Recently reversed moves are forbidden (tabu), unless they give a new best solution (the aspiration rule).
- **Simulated Annealing**: accept a worse move with probability *exp(−Δ/T)*. The temperature *T* is lowered gradually and can be raised again to escape a local optimum.
- **Genetic Algorithm**: evolve a population of solutions using selection, crossover (e.g., order crossover on a giant tour), and mutation.
- **Hybrid Genetic Search (HGS)**: a genetic algorithm combined with:
  - local-search improvement of every child
  - the Split decoder, which cuts a giant tour into routes
  - separate feasible and infeasible subpopulations
  - fitness that rewards both low cost and diversity

  HGS is state of the art for CVRP.

## Key academic references

| Reference | DOI |
|---|---|
| Dantzig, G. B., & Ramser, J. H. (1959). The Truck Dispatching Problem. *Management Science*, 6(1), 80–91. | [10.1287/mnsc.6.1.80](https://doi.org/10.1287/mnsc.6.1.80) |
| Clarke, G., & Wright, J. W. (1964). Scheduling of Vehicles from a Central Depot to a Number of Delivery Points. *Operations Research*, 12(4), 568–581. | [10.1287/opre.12.4.568](https://doi.org/10.1287/opre.12.4.568) |
| Held, M., & Karp, R. M. (1962). A Dynamic Programming Approach to Sequencing Problems. *J. SIAM*, 10(1), 196–210. | [10.1137/0110015](https://doi.org/10.1137/0110015) |
| Gillett, B. E., & Miller, L. R. (1974). A Heuristic Algorithm for the Vehicle-Dispatch Problem. *Operations Research*, 22(2), 340–349. | [10.1287/opre.22.2.340](https://doi.org/10.1287/opre.22.2.340) |
| Beasley, J. E. (1983). Route first—Cluster second methods for vehicle routing. *Omega*, 11(4), 403–408. | [10.1016/0305-0483(83)90033-6](https://doi.org/10.1016/0305-0483(83)90033-6) |
| Osman, I. H. (1993). Metastrategy simulated annealing and tabu search algorithms for the vehicle routing problem. *Annals of Operations Research*, 41, 421–451. | [10.1007/BF02023004](https://doi.org/10.1007/BF02023004) |
| Gendreau, M., Hertz, A., & Laporte, G. (1994). A Tabu Search Heuristic for the Vehicle Routing Problem. *Management Science*, 40(10), 1276–1290. | [10.1287/mnsc.40.10.1276](https://doi.org/10.1287/mnsc.40.10.1276) |
| Prins, C. (2004). A simple and effective evolutionary algorithm for the vehicle routing problem. *Computers & Operations Research*, 31(12), 1985–2002. | [10.1016/S0305-0548(03)00158-8](https://doi.org/10.1016/S0305-0548(03)00158-8) |
| Vidal, T., Crainic, T. G., Gendreau, M., Lahrichi, N., & Rei, W. (2012). A Hybrid Genetic Algorithm for Multidepot and Periodic Vehicle Routing Problems. *Operations Research*, 60(3), 611–624. | [10.1287/opre.1120.1048](https://doi.org/10.1287/opre.1120.1048) |
| Toth, P., & Vigo, D. (Eds.) (2014). *Vehicle Routing: Problems, Methods, and Applications* (2nd ed.). SIAM. | [10.1137/1.9781611973594](https://doi.org/10.1137/1.9781611973594) |
| Uchoa, E., et al. (2017). New benchmark instances for the Capacitated Vehicle Routing Problem. *European Journal of Operational Research*, 257(3), 845–858. | [10.1016/j.ejor.2016.08.012](https://doi.org/10.1016/j.ejor.2016.08.012) |
| Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood. *Computers & Operations Research*, 140, 105643. | [10.1016/j.cor.2021.105643](https://doi.org/10.1016/j.cor.2021.105643) |
| Wouda, N. A., & Lan, L. (2023). ALNS: a Python implementation of the adaptive large neighbourhood search metaheuristic. *JOSS*, 8(81), 5028. | [10.21105/joss.05028](https://doi.org/10.21105/joss.05028) |
| Wouda, N. A., Lan, L., & Kool, W. (2024). PyVRP: A High-Performance VRP Solver Package. *INFORMS Journal on Computing*, 36(4), 943–955. | [10.1287/ijoc.2023.0055](https://doi.org/10.1287/ijoc.2023.0055) |

## License

This project is released under the MIT License. See [LICENSE](LICENSE). Copyright (c) 2026 Abdulkaream KA.

The repositories listed above belong to their authors and have their own licenses.
