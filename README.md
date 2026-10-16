# CoRectron: contextual recommendation experiments

Implementation and synthetic experiments for **Simple Projection-Free Algorithm
for Contextual Recommendation with Logarithmic Regret and Robustness**, by
Shinsaku Sakaue. The code is distributed under the [MIT License](LICENSE).

The package contains CoRectron-L, kernel CoRectron, ONS, kernel ONS (KONS), and
OGD; linear and kernel data generators; regularization sweeps; and experiments
with suboptimal user feedback.

## Install and test

Use Python 3.11 and create an environment in the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

`requirements-dev.txt` installs the pinned experiment dependencies and pytest.
For experiments without the tests, install `requirements-validated.txt` instead.
`requirements.txt` provides looser installation constraints; those lower bounds
are not a claim that every allowed version combination has been tested.

The pinned numerical packages are NumPy 2.4.3, pandas 3.0.1, SciPy 1.17.1,
Matplotlib 3.10.8, and PyYAML 6.0.3. The paper experiments used Python 3.11.16 on
a MacBook Air with an Apple M4 chip (10 cores), 32 GB RAM, and macOS 26.5.2.
NumPy and SciPy used Apple Accelerate. Wall-clock measurements depend on the
interpreter, numerical libraries, hardware, and other workloads.

## Smoke run

For direct Python commands, fix numerical-library thread counts before starting
Python:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
MPLCONFIGDIR=/tmp/corectron-mpl .venv/bin/python run_experiment.py \
  --config configs/smoke_linear.yaml --output results/smoke_check
MPLCONFIGDIR=/tmp/corectron-mpl .venv/bin/python plot_results.py \
  --input results/smoke_check --output figures/smoke_check
```

The smoke configuration has 300 rounds and three seeds. Its
`aggregate_summary.csv` contains `corectron`, `ons`, and `ogd`.

## Comparison and feedback experiments

Run the linear and RBF-kernel regularization sweeps:

```bash
bash scripts/run_main_campaign_unattended.sh comparison_run \
  0.001,0.01,0.1,1,10,100,1000
```

Run the suboptimal-feedback sweeps:

```bash
bash scripts/run_subopt_campaign_unattended.sh feedback_run
```

The launchers use `.venv/bin/python` by default and set all five thread variables
above to 1. Set `PYTHON_BIN=/absolute/path/to/python` to select another environment.
Use a new run identifier for a new code version or experiment configuration.
See [the reproduction guide](docs/REPRODUCTION.md) for settings, outputs,
validated resume behavior, and interpretation of the measurements.

The comparison design has 126 configurations and ten seeds per configuration:
21 linear configurations and 105 kernel configurations. The feedback design has
66 configurations and ten seeds per configuration. Runtimes vary by machine and
by the frequency of projections; no fixed completion time is assumed.

## Paper data and figures

The per-seed summaries and resolved configurations for the paper's 192 settings
are included in `data/paper/`, together with checksums and the measured source
version. Regenerate Figures 1–5 without rerunning experiments:

```bash
.venv/bin/python scripts/build_paper_figures.py --out-root figures/paper
```

For newly generated results, use:

```bash
.venv/bin/python scripts/build_paper_figures.py \
  --comparison-root results/comparison_run --feedback-root results/feedback_run \
  --out-root figures/new_run
```

See [the reproduction guide](docs/REPRODUCTION.md) for the figure mapping and
numerical reproducibility limits. The source version used for the measurements
is distinct from later changes to documentation, launchers, and plotting tools.
The reproduction guide also describes the projection implementation and its
validation.

Tests, configurations, plotting scripts, and reproduction instructions are
included. Newly generated results, figures, and caches stay outside version
control; the small paper-data bundle is included explicitly.
