# Presentation notes

## Evidence to put on slides

| Slide | Use this file |
|---|---|
| Benchmark A / B / P | `results/<name>/console_output.png` + `routes.png` (and your own live screenshot, see below) |
| Validation | `results/<name>/validation.txt` (shows "VALID SOLUTION: YES") |
| Convergence | `results/<name>/convergence.png` |
| Step-by-step evidence | `results/<name>/trace.txt` (savings merges, constructions, every new best and which operator found it) |
| v2.0 vs reference-repo methods | `results/comparison_reference_methods.csv` |
| Scaling + ablation | `results/ablation_x_instances.csv` |
| Robustness | `results/robustness_seeds.csv` (10/10 seeds reach the optimum) |
| Generalization | `results/generalisation.csv` |
| Self-created case | `results/SCG-CPAC-Narathiwat-25/routes.png`, `..._co2/routes.png`, `own_case_comparison.csv` |

## Taking the required screenshots yourself

The PNG files are generated from the real output. The organizers will likely expect actual screenshots too:

1. Open a terminal in the project folder (VS Code terminal is fine).
2. Run `python solver.py benchmarks/A-n32-k5.vrp benchmarks/B-n31-k5.vrp benchmarks/P-n40-k5.vrp --time 30 --seed 1`
3. Take a screenshot of each result block (Win + Shift + S), with the command and the clock visible.
4. Run `python validate.py benchmarks/A-n32-k5.vrp results/A-n32-k5/solution.sol` (and the same for B and P) and take a screenshot.
5. Do not edit any output file. If you need the numbers again, run the solver again.

## Questions judges may ask

- "Where is the AI?" → Two parts:
  - **Evolutionary computation:** the hybrid genetic search evolves a population of route plans.
  - **Online learning:** a UCB1 bandit learns during the run which mutation pays off on this instance. Its table is in `result.json → engine_statistics.bandit`.
- "Did you use a library?" → No. Every algorithm is hand-written with the Python standard library (`MANUAL_METHODS.md`). The formulas are checked against brute force in `tests/`.
- "Is the answer really optimal?" → On A/B/P it matches the published optimum. The report also shows that every route's order is proven optimal by Held-Karp, and the set-partitioning step finished, so the route pool contains no better combination.
- "Did you hard-code 784?" → No. Show `cvrp_parser.py`: it only reads the file. The optimum is read from the file comment after solving, only to print the gap. Run the solver on one of the extra instances, or change a demand in a copy of the file, and it will still produce a new valid answer.
- "Why is the benchmark solved so fast?" → 30–40 customers is small for a hybrid genetic search. The fairer tests are:
  - the 8 extra instances: all optimal
  - the X instances with 100–200 customers: gaps of 0.24–0.74% in 45 s
- "How do you guarantee capacity?" → The Split decoder limits the number of routes to K. Any overload is penalized and then repaired. The final check in `evaluation.py` and `validate.py` rejects any overloaded route.
- "What is green about it?" → The `--objective co2` mode uses load-dependent fuel use. On the Narathiwat case it cuts CO2 by 28.4% compared with a manual-style plan, and by 2.1% compared with the distance-optimal plan.
- "Limitations?" → Euclidean distance × 1.3 instead of real roads, and no time windows yet. The fuel coefficients are assumptions.
