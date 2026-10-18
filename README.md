# Cheap Search, Costly Holds

**Reservation Synchronization in Agentic Markets**
Shivam Gupta, Independent Researcher, India (2026)

This artifact studies temporary exclusive reservations in delegated shopping. It contains a technical manuscript, executable economic model, finite continuous-time Markov chains, a separate exact rational oracle, physical-slot simulations, and all results used in the paper.

The experiments are constructed markets, not marketplace observations or LLM evaluations. Their purpose is to verify a conditional economic mechanism and test the limits of an independence approximation. No acceptance or deployment claim is implied.

## Main results

- Full-market reservation attempts have identical one-supplier pre-commitment capacity processes under common and independent exponential release. Distinct successful-mandate probabilities differ by `(1 + lambda*M*tau)/(1 + lambda*tau)`.
- In an independent-availability model, holding duration growing as `a*k**beta` creates a cheap-search limit with completion proportional to `c**(beta/(beta+1))`. Constant duration gives a positive limit.
- All symmetric finite-menu equilibria are enumerated using pure actions and adjacent-action indifference roots. Multiple equilibria are retained.
- A scalar family of LPs gives a feasible planner policy and an interval upper bound; the code reports the remaining objective gap. A supporting capacity-time toll is verified on the reported cases.
- A deterministic-release discrepancy in short simulations largely disappears with longer warm-up. The exploratory diagnosis is retained, rather than presenting the initial gap as stationary evidence.

## Reproduce

Python 3.12 was used. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
python verify_results.py
python check_welfare_asymptotics.py
python run_experiments.py
python diagnose_fixed.py
python make_figures.py
tectonic paper/paper.tex --outdir paper --keep-logs
```

The complete main experiment used four processes and took about 6.4 minutes on the original machine; runtime varies. It records 840 runs and 103.2 million counted arrivals (120 million including warm-up). The separately identified post-hoc horizon diagnostic adds 36 runs and 18 million counted arrivals. `verify_results.py` checks already saved results without rerunning the simulations. The experiment plan was frozen after model development and a pilot; it is not a preregistration.

`requirements.txt` pins the versions used. Matplotlib font rendering and LP details can vary across operating systems. Numerical tolerances and exact arithmetic are distinguished below. Tectonic is needed only to compile the paper, and Poppler is useful for visual inspection.

## Files

| Path | Purpose |
|---|---|
| `src/holdmarket.py` | Economic model, equilibrium enumeration, globally bounded planner, CTMC, and physical-slot simulator |
| `tests/test_model.py` | Independent labeled-state rational oracle and implementation checks |
| `experiment_plan.json` | Frozen main specification and all main seeds |
| `run_experiments.py` | Analytical calculations and 840 main simulation runs |
| `diagnose_fixed.py` | Explicitly exploratory horizon sensitivity after the main run |
| `results/provenance.json` | Main model-source hash, plan hash, Python version, and data classification |
| `results/*_runs.json` | Every main and diagnostic run-level record |
| `results/finite_exact.json` | 231 finite CTMC cases, including residuals |
| `verify_results.py` | Independently recomputed aggregation and result integrity checks |
| `make_figures.py` | All five vector figures and three manuscript tables |
| `paper/paper.tex`, `paper/paper.pdf` | Source and full manuscript |
| `notes/literature_review.txt` | Search scope, nearby work, and contribution boundaries |

## Interpretation and numerical scope

`success` records a mandate obtaining at least one hold at arrival. One eventual transaction per successful mandate is assumed; there is no modeled subsequent execution failure. The simulator is not an end-to-end purchase agent. It retains concrete supplier slots, uses distinct supplier samples, and rejects double releases. Common exponential, common deterministic, and independent exponential release are separate mechanisms.

Marginal availability conservation is exact at fixed breadth with zero post-use occupation, but `1-(1-q)**k` requires independent availability. The paper does not prove this approximation's general large-market convergence. Full-market breadth demonstrates its failure. Fixed-clock observations must be interpreted at their recorded horizons.

The planner's interval relaxation supplies a mathematical upper-bound construction evaluated with floating-point SciPy/HiGHS LPs; it is not an exact rational certificate for arbitrary real inputs. The small labeled-state stationarity oracle uses Python `Fraction` and exact Gaussian elimination. The checked grid contains 15 oracle comparisons and 216 economic parameter configurations. Seven test functions include a journal-stage terminal-partition audit of the planner upper bound. Monte Carlo intervals describe precision over simulation seeds, not uncertainty about a real economy.

## License and citation

Code is available under the MIT license. Manuscript and original figures are CC BY 4.0. External papers are referenced, not redistributed. For scholarly use, cite:

> Gupta, Shivam. 2026. *Cheap Search, Costly Holds: Reservation Synchronization in Agentic Markets.* Independent research manuscript. https://github.com/shi1720/cheap-search-costly-holds

## Journal replication update (18 October 2026)

The journal replication release preserves `paper/` as the original public working paper. The journal-formatted manuscript is supplied separately to the editor. This update corrects the planner's reported bound by retaining upper bounds from pruned terminal intervals. A terminal-partition regression fails on the old routine and passes after correction. The three central feasible policies and welfare values are unchanged.

All 840 main simulation records and their aggregate summaries were repeated with the original seeds and match the previous records byte for byte. The separate exploratory horizon diagnostic is also repeated and checked independently. `check_welfare_asymptotics.py` provides an 80-digit Decimal recipe for the previously recorded seven beta-one welfare checks. The output is a high-precision numerical check, not an exact-arithmetic certificate.

`notes/journal_replication.txt` and the replication evidence files distinguish the corrections, repeated computations and interpretation limits. The historical release `v1.0.0` remains unchanged.
