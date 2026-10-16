# Reproducing the experiments

Run commands from the repository root after following the installation and test
steps in [README.md](../README.md).

## Experiment design

| Experiment | Rounds per seed | Seeds | Settings |
|---|---:|---|---|
| Linear comparison | 10000 | 0–9 | 3 methods × 7 regularization coefficients |
| RBF-kernel comparison | 1000 | 0–9 | 5 methods × 7 coefficients × 3 bandwidths |
| Suboptimal feedback | 10000 | 0–9 | 3 methods × 11 levels × 2 feedback models |

The coefficient grid is `0.001,0.01,0.1,1,10,100,1000`; the RBF bandwidths are
`0.5,1,2`. The action and context dimensions are both 10, and an action selects
5 items. The generator clips Gaussian contexts to the configured context ball.
It normalizes the linear utility matrix in Frobenius norm or the kernel utility
function in RKHS norm; the supplied configurations set these norms and the
context bound to one.

Each seed fixes the contexts, true utilities, and feedback randomness shared by
the methods. Feedback generation is independent of the method order. Ties in
action scores are resolved by a stable ascending sort followed by the last
`m` indices, so larger indices are selected first among ties.

The linear methods are `corectron`, `ons`, and `ogd`. The kernel comparison also
includes `corectron_rbf` and `kons`; `corectron_rbf` is the existing identifier for
kernel CoRectron, including when a polynomial kernel is selected. The linear
methods in kernel-generated data use linear predictions as misspecified
baselines.

## Regularization campaign

```bash
bash scripts/run_main_campaign_unattended.sh comparison_run \
  0.001,0.01,0.1,1,10,100,1000
```

The optional third positional argument `1` adds polynomial-kernel experiments
with degrees `2,4,8`. They are outside the linear/RBF comparison design above.
The launchers choose `.venv/bin/python`, or the executable named by `PYTHON_BIN`.
On macOS, an optional `caffeinate -dimsu` prefix prevents sleep during a run.

Outputs include:

- `results/comparison_run/linear_t10000_pow10_s10/`
- `results/comparison_run/kernel_rbf_sigma*_t1000_pow10_s10_all5/`
- `results/comparison_run/campaign_status_latest.tsv`
- `figures/comparison_run/dashboards/curves_dashboard_linear.png`
- `figures/comparison_run/dashboards/spread_dashboard_linear.png`
- `figures/comparison_run/dashboards/curves_dashboard_kernel_rbf.png`
- `figures/comparison_run/dashboards/spread_dashboard_kernel_rbf.png`

To plot existing comparison summaries without rerunning learners:

```bash
.venv/bin/python scripts/build_sweep_dashboards.py \
  --run-root results/comparison_run \
  --out-root figures/comparison_run/dashboards --groups linear,kernel_rbf
```

The comparison plots show every coefficient in the sweep. The final-regret
annotations in the spread plots give the lowest mean for each method and
setting. The feedback experiment uses coefficients selected from the same ten
optimal-feedback seeds. Shaded bands in the curve plots are pointwise Student-t
95% intervals over seeds; they are not simultaneous intervals over the grid or
selection-adjusted statements of statistical superiority.

## Suboptimal feedback

```bash
bash scripts/run_subopt_campaign_unattended.sh feedback_run
```

The optional second and third positional arguments override the one-swap and
relative score-noise grids; both default to `0.0,0.1,...,1.0`.
The default regularization coefficients are `0.001` for CoRectron-L, `0.1` for
ONS, and `1.0` for OGD, selected from the optimal-feedback sweep used in the paper.
They are fixed inputs to the feedback experiment, not newly selected at each
noise level. They can be overridden through the combined script's
`--best-coeff-*` options.

A smaller check is:

```bash
MPLCONFIGDIR=/tmp/corectron-mpl .venv/bin/python scripts/run_subopt_combined.py \
  --config configs/smoke_linear.yaml \
  --subopt-prob-grid 0.0,0.5,1.0 --subopt-noise-std-grid 0.0,0.5,1.0 \
  --output-root results/smoke_feedback --figure-root figures/smoke_feedback
```

The separate `run_subopt_sweep.py` and `run_subopt_noise_sweep.py` entry points
produce per-seed regret-versus-suboptimality plots for either model. Their
default grids and methods match the combined script; the noise parameter is
relative to the current utility-vector norm. The probability script also
supports the explicitly selected `--subopt-mode uniform` alternative.

The runner saves per-configuration results beneath `one_swap/` and
`score_perturb/`, plus `one_swap_summary.csv` and `score_perturb_summary.csv`.
The plot is `figures/feedback_run/subopt_combined.png`.

`final_regret` compares recommendations with the revealed user actions.
`cumulative_suboptimality` is the summed utility gap between the optimal and
revealed actions. Their sum is regret relative to optimal actions. A lower
revealed-action regret at a higher noise level need not imply better
recommendations relative to optimal actions. The CSV files retain both
components for analysis.

## Paper data and figure generation

`data/paper/` contains all 126 comparison and 66 feedback settings, their
resolved configurations, and per-seed summaries. It includes no per-round traces.
The data manifest identifies the source revision used for measurement, its
source-file hashes, the measured environment, and data-file checksums. This
identity is preserved when packaging and plotting code are updated.

Public revision `b630751b756fe23a986e0b8b1901011e77aea9c6` contains the same
numerical helpers (`experiments_core/utils.py`) as that measurement source.
Later fixes to kernel input handling and tiny-vector clipping are not reflected
in the bundled runtime measurements.

```bash
.venv/bin/python scripts/build_paper_figures.py --out-root figures/paper
```

| Figure | Output file |
|---|---|
| 1 | `spread_dashboard_linear.png` |
| 2 | `spread_dashboard_kernel_rbf.png` |
| 3 | `curves_dashboard_linear.png` |
| 4 | `curves_dashboard_kernel_rbf.png` |
| 5 | `subopt_combined.png` |

To plot new comparison and feedback campaigns together:

```bash
.venv/bin/python scripts/build_paper_figures.py \
  --comparison-root results/comparison_run --feedback-root results/feedback_run \
  --out-root figures/new_run
```

The paper measurements used CPython 3.11.16, the pinned package versions in
`requirements-validated.txt`, and Apple Accelerate on macOS 26.5.2 (Apple M4,
10 cores, 32 GB RAM), with AC power connected. Matching package versions on
another platform does not guarantee identical floating-point results or timings.
Small score differences can change the selected action when scores are tied or
nearly tied; rounding near the domain boundary can also change projection counts.
The bundled summaries retain the values underlying the paper figures.

## Run identity and resuming

Each launcher fixes the campaign design, configuration-file hashes, source
hashes, and environment in `campaign_manifest.json` before running any jobs.
Reusing an identifier with different inputs is rejected before execution. Logs,
status tables, and environment snapshots are kept separately for each invocation
under `invocations/`; `campaign_status_latest.tsv` links to the latest status.
The launchers stop on the first failure and never automatically retry. Completed
matching settings can be reused by rerunning the same command after the failure
has been inspected. Partial settings still require the explicit recovery below.

A `.campaign-lock` directory prevents overlapping launcher invocations. After an
interrupted launch, verify that its processes have exited before removing a
stale lock. A campaign directory from a version without `campaign_manifest.json`
is preserved; use a new identifier rather than silently adopting it.

A run saves its configuration in `resolved_config.yaml` and its configuration
hash, source-file hashes, Python and package versions, platform, and thread
settings in `manifest.json`. `SUCCESS` records the output checksums. The result
files include per-seed and aggregate summaries;
`round_metrics.csv` is written when `runtime.save_round_metrics` is true.

Resume requires a completed run with a matching configuration, source and
environment identity, and matching output checksums. An incompatible or partial
output is rejected; use a new output directory, or explicitly use `--force` to
replace that run. A legacy directory without the required metadata is not a
validated cache. Preserve historical directories when comparing versions.

`SUCCESS` marks completion, `RUNNING` marks an active attempt, and `FAILED`
records a failed attempt. A leftover `RUNNING` marker blocks reuse even with
`--force`; first check that no process owns the directory, then remove the stale
marker before retrying. Numerical failures raise an error rather than
silently skipping updates or substituting a different projection.

## Projection implementation and validation

ONS projects onto the Euclidean unit ball in its curvature metric. KONS uses
the corresponding RKHS curvature metric through a coefficient representation,
with the RKHS unit ball as its constraint. The projection error tolerance is
`1e-10` in the Euclidean norm for ONS and in the RKHS norm for KONS.

The tests compare KONS projections with an independent finite-feature primal
solution, including rank-deficient feature sets, and compare linear-kernel KONS
with finite-dimensional ONS. Numerical input and accuracy checks reject invalid
updates or unsuccessful solves instead of silently substituting another update.

Kernel helpers accept finite real integer or floating-point arrays and compute
in float64. They reject nonnumeric, Boolean, and complex arrays. A coordinate
difference that cannot be represented in float64 raises an error; a representable
RBF difference with an extremely large scaled distance may yield a zero kernel
value. Polynomial-kernel intermediate overflow also raises an error.

## Runtime measurement

Each round's timer covers score prediction, linear optimization for the
recommendation, and the learner update. Synthetic data and user-action
generation, serialization, and plotting are outside this timer. Both wall-clock
seconds and process CPU seconds are recorded; the comparison plots use
wall-clock seconds.

The launchers set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS` to 1 before Python starts.
Set these variables yourself for direct Python commands. Compare timings only
under matched interpreter, numerical-library, hardware, and thread conditions.

CoRectron-L maintains a dense inverse through a rank-one update; ONS maintains a
Cholesky factor. Their measured runtime difference includes these implementation
choices as well as any projection work. The projection-event counter records
rounds requiring a nontrivial projection; it is not a measurement of time spent
in projection. Quantifying the fraction of runtime attributable to projection would require
a separate timing breakdown or a controlled implementation comparison.

The plotting tools use Matplotlib's noninteractive `Agg` backend. Generated
results and plots stay under the ignored `results/` and `figures/` directories.
