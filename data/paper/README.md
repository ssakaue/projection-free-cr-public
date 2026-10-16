# Data for Figures 1–5

These are the final measurements used in the paper: 192 parameter settings,
10 seeds per setting (0–9), and 1,920 seed runs in total. The CSV and resolved
configuration files are unchanged from those runs. `manifest.json` gives their
SHA-256 checksums, the source version used for the experiments, package versions,
linear-algebra backends, and thread settings. Plotting and packaging changes after
the experiments do not change that source provenance.

## Contents

- `comparison/`: one linear scenario and three RBF-kernel scenarios, containing
  126 settings. Each scenario has `sensitivity_summary.csv`; each setting has
  `seed_summary.csv` and `resolved_config.yaml`.
- `feedback/`: one-swap and relative score-perturbation sweeps, containing
  66 settings. The two combined summaries contain individual seed measurements;
  each setting also has its complete seed summary and resolved configuration.

The linear and feedback runs have 10,000 rounds each. Kernel runs have 1,000
rounds each. Full round-by-round logs are not needed for Figures 1–5 and are not
included; all final measurements, parameter settings, and seeds needed to
recompute the plotted statistics are included.

## Recreate the figures

From the repository root, after creating `.venv` and installing dependencies as
described in the repository README:

```sh
.venv/bin/python scripts/build_paper_figures.py --validate-only
.venv/bin/python scripts/build_paper_figures.py --out-root figures/paper
```

These commands do not run experiments. Before plotting, the script checks the
bundled checksums, seed coverage, resolved parameters, and agreement of the
summary statistics with the individual seed measurements. It then writes:

| Figure | File |
| --- | --- |
| 1 | `spread_dashboard_linear.png` |
| 2 | `spread_dashboard_kernel_rbf.png` |
| 3 | `curves_dashboard_linear.png` |
| 4 | `curves_dashboard_kernel_rbf.png` |
| 5 | `subopt_combined.png` |

To plot a newly completed full set of experiments, supply its two result roots:

```sh
.venv/bin/python scripts/build_paper_figures.py \
  --comparison-root results/comparison \
  --feedback-root results/feedback \
  --out-root figures/new-run
```

The comparison root must contain the four scenario directories. The feedback
root must contain both combined summaries and the `one_swap/` and
`score_perturb/` setting directories. The data checks still apply; the bundled
checksums apply only when using `--data-root` (or its default). New results must
use the same dataset, normalization, learner parameters, and feedback policy as
the paper experiments; progress messages and round-log output settings may differ.

## Measurements and uncertainty

`final_regret` is cumulative regret relative to the actual user actions, including
suboptimal user actions in the feedback sweeps. `cumulative_suboptimality` is the
cumulative utility difference between optimal and actual user actions.
`total_runtime_sec` measures prediction, recommendation optimization, and learner
updates with `time.perf_counter`; data and user-action generation are excluded.
`total_cpu_time_sec` uses `time.process_time`. `projection_events` counts rounds
requiring a nontrivial Mahalanobis projection.

In comparison summaries, `coeff` is the regularization coefficient. `reg_value`
is the CoRectron regularizer, the ONS/KONS curvature regularizer, or the inverse
OGD step size, depending on `algorithm`. Each resolved configuration gives the
actual values. Feedback summaries use `subopt_prob` for the one-swap probability
and `noise_std` for the relative score-noise level.

Each plotted point is the arithmetic mean across the 10 seeds. The bands in
Figures 3–5 are pointwise 95% Student-t confidence intervals, using the sample
standard deviation (9 degrees of freedom), divided by the square root of 10,
and multiplied by the 97.5th percentile of the t distribution with 9 degrees of
freedom. Figures 1–2 show the means for each coefficient without intervals.

Exact pixels depend on the available fonts and Matplotlib version. The original
plots used Matplotlib 3.10.8 and Times New Roman on macOS; the scripts fall back
to other serif fonts when needed. Repeating the experiments can also change
wall-clock timing and floating-point tie-breaking across numerical environments.
