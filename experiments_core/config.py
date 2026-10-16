from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .action_sets import build_action_set
from .utils import to_pretty_json
from .validation import finite_float, finite_tree, integer


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def _parse_seeds_arg(raw: str) -> list[int]:
    if not raw:
        raise ValueError("--seeds is empty")
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("--seeds is empty")
    return values


def resolve_config(
    raw: dict[str, Any],
    cli_ons_lr: float | None,
    cli_horizon: int | None,
    cli_seeds: str | None,
    cli_ons_epsilon: float | None = None,
    cli_corectron_lambda: float | None = None,
    cli_ogd_step0: float | None = None,
    cli_kons_lr: float | None = None,
    cli_corectron_rbf_lambda: float | None = None,
    cli_kons_epsilon: float | None = None,
    cli_kernel_type: str | None = None,
    cli_kernel_sigma: float | None = None,
    cli_kernel_degree: int | None = None,
    cli_kernel_coef0: float | None = None,
    cli_kernel_alpha: float | None = None,
    cli_subopt_prob: float | None = None,
    cli_subopt_mode: str | None = None,
    cli_subopt_noise_std: float | None = None,
) -> dict[str, Any]:
    for name, value in {"ons_lr": cli_ons_lr, "ons_epsilon": cli_ons_epsilon,
                        "corectron_lambda": cli_corectron_lambda, "ogd_step0": cli_ogd_step0,
                        "kons_lr": cli_kons_lr, "corectron_rbf_lambda": cli_corectron_rbf_lambda,
                        "kons_epsilon": cli_kons_epsilon, "kernel_sigma": cli_kernel_sigma,
                        "kernel_alpha": cli_kernel_alpha}.items():
        if value is not None:
            finite_float(value, name, positive=True)
    for name, value in {"kernel_coef0": cli_kernel_coef0, "subopt_prob": cli_subopt_prob,
                        "subopt_noise_std": cli_subopt_noise_std}.items():
        if value is not None:
            finite_float(value, name, nonnegative=True)
    cfg = deepcopy(raw)
    finite_tree(cfg)
    for section in ("experiment", "dataset", "learners"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"{section} must be a mapping")
    for name, value in cfg["learners"].items():
        if name not in {"corectron", "ons", "ogd", "corectron_rbf", "kons"}:
            raise ValueError(f"Unknown learner: {name}")
        if not isinstance(value, dict) or not isinstance(value.get("enabled", False), bool):
            raise ValueError(f"learners.{name} must be a mapping with a boolean enabled value")
        for key, item in value.items():
            if key not in {"enabled", "schedule", "kernel_type", "true_comparator_in_domain"} and item is not None:
                finite_float(item, f"learners.{name}.{key}")
    if not any(item.get("enabled", False) for item in cfg["learners"].values()):
        raise ValueError("At least one learner must be enabled")
    runtime = cfg.setdefault("runtime", {})
    if not isinstance(runtime, dict) or not isinstance(runtime.get("save_round_metrics", True), bool):
        raise ValueError("runtime.save_round_metrics must be boolean")

    if cli_horizon is not None:
        cfg["experiment"]["horizon"] = integer(cli_horizon, "--horizon", minimum=1)
    if cli_seeds is not None:
        cfg["experiment"]["seeds"] = _parse_seeds_arg(cli_seeds)

    horizon = integer(cfg["experiment"]["horizon"], "experiment.horizon", minimum=1)
    cfg["experiment"]["horizon"] = horizon
    seeds = cfg["experiment"].get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("experiment.seeds must be a nonempty list")
    seeds = [integer(seed, "experiment.seeds") for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("experiment.seeds must be unique")
    cfg["experiment"]["seeds"] = seeds
    cfg["experiment"]["progress_every"] = integer(cfg["experiment"].get("progress_every", 0), "experiment.progress_every")

    dcfg = cfg["dataset"]
    lcfg = cfg["learners"]

    ncfg = cfg.get("normalization", {})
    if not isinstance(ncfg, dict):
        raise ValueError("normalization must be a mapping when provided")
    if ncfg.get("enabled", True) is not True:
        raise ValueError("Normalization is mandatory. Set normalization.enabled=true.")

    policy = str(ncfg.get("policy", "set_level"))
    if policy != "set_level":
        raise ValueError("Only normalization.policy='set_level' is supported.")

    target_action_residual_l2 = finite_float(ncfg.get("target_action_residual_l2", 1.0), "normalization.target_action_residual_l2", positive=True)
    target_context_l2 = finite_float(ncfg.get("target_context_l2", 1.0), "normalization.target_context_l2", positive=True)
    if target_action_residual_l2 <= 0.0:
        raise ValueError("normalization.target_action_residual_l2 must be positive")
    if target_context_l2 <= 0.0:
        raise ValueError("normalization.target_context_l2 must be positive")

    action_dim = integer(dcfg["action_dim"], "dataset.action_dim", minimum=1)
    context_dim = integer(dcfg["context_dim"], "dataset.context_dim", minimum=1)
    m_select = integer(dcfg["m_select"], "dataset.m_select", minimum=1)
    raw_context_norm = finite_float(dcfg.get("context_norm_raw", dcfg.get("context_norm", 1.0)), "dataset.context_norm", positive=True)
    if raw_context_norm <= 0.0:
        raise ValueError("dataset.context_norm must be positive")

    action_set_raw = build_action_set(action_dim=action_dim, m_select=m_select, scale=1.0)
    bounds_raw = action_set_raw.residual_bounds()
    residual_l2_max_raw = float(bounds_raw.residual_l2_max)
    if residual_l2_max_raw <= 0.0:
        raise ValueError("Action-set residual l2 bound must be positive")

    action_scale = target_action_residual_l2 / residual_l2_max_raw
    context_scale = target_context_l2 / raw_context_norm

    action_set = build_action_set(action_dim=action_dim, m_select=m_select, scale=action_scale)
    bounds = action_set.residual_bounds()
    residual_l2_max = float(bounds.residual_l2_max)
    context_norm = raw_context_norm * context_scale

    dim_h_linear = action_dim * context_dim
    d_ref = max(1, int(dim_h_linear))

    # In normalized setting, linear and kernel gradient scales are both O(1).
    g_bound_linear = residual_l2_max * context_norm

    dcfg["action_set_type"] = action_set.name
    dcfg["action_scale"] = action_scale
    dcfg["context_scale"] = context_scale
    dcfg["context_norm_raw"] = raw_context_norm
    dcfg["context_norm"] = context_norm
    dcfg["residual_l2_max_raw"] = residual_l2_max_raw
    dcfg["residual_l2_max"] = residual_l2_max
    dcfg["dim_h_linear"] = dim_h_linear
    dcfg["d_ref"] = d_ref
    dcfg["g_bound_linear"] = g_bound_linear
    cfg["dataset"] = dcfg

    ncfg["enabled"] = True
    ncfg["policy"] = policy
    ncfg["target_action_residual_l2"] = target_action_residual_l2
    ncfg["target_context_l2"] = target_context_l2
    ncfg["action_scale"] = action_scale
    ncfg["context_scale"] = context_scale
    ncfg["residual_l2_raw"] = residual_l2_max_raw
    ncfg["context_l2_raw"] = raw_context_norm
    ncfg["residual_l2_normalized"] = residual_l2_max
    ncfg["context_l2_normalized"] = context_norm
    cfg["normalization"] = ncfg

    # Payoff bounds include the configured true-utility and feature norms.
    # The kernel regularization reference below uses d_ref as an experimental
    # dimension proxy; it does not assert a finite-dimensional kernel theorem.
    model_type = str(dcfg.get("model_type", "linear_contextual"))
    if model_type not in {"linear_contextual", "kernel_contextual"}:
        raise ValueError("dataset.model_type must be linear_contextual or kernel_contextual")
    linear_cfg = dcfg.get("linear", {})
    kernel_cfg_input = dcfg.get("kernel", {})
    if not isinstance(linear_cfg, dict) or not isinstance(kernel_cfg_input, dict):
        raise ValueError("dataset.linear and dataset.kernel must be mappings")
    linear_true_norm = finite_float(linear_cfg.get("true_weight_norm", 1.0), "dataset.linear.true_weight_norm", positive=True)
    kernel_true_norm = finite_float(kernel_cfg_input.get("true_weight_norm", 1.0), "dataset.kernel.true_weight_norm", positive=True)
    selected_kernel = str(cli_kernel_type if cli_kernel_type is not None else kernel_cfg_input.get("kernel_type", "rbf")).lower()
    selected_degree = integer(cli_kernel_degree if cli_kernel_degree is not None else kernel_cfg_input.get("degree", 2), "dataset.kernel.degree", minimum=1)
    kernel_feature_bound = 1.0 if selected_kernel == "rbf" else context_norm ** selected_degree
    true_score_bound = context_norm * linear_true_norm if model_type == "linear_contextual" else kernel_feature_bound * kernel_true_norm
    core_cfg = lcfg.get("corectron", {"enabled": False})
    b_payoff = finite_float(residual_l2_max * true_score_bound, "payoff bound", positive=True)
    lambda_lin_theory = b_payoff * b_payoff * float(d_ref)
    core_cfg["b_payoff"] = b_payoff
    core_cfg["lambda_base"] = float(d_ref)
    core_cfg["lambda_reg_theory"] = lambda_lin_theory
    core_raw = core_cfg.get("lambda_reg")
    core_cfg["lambda_reg"] = (
        float(cli_corectron_lambda)
        if cli_corectron_lambda is not None
        else float(lambda_lin_theory if core_raw is None else core_raw)
    )
    if core_cfg["lambda_reg"] <= 0.0:
        raise ValueError("corectron.lambda_reg must be positive")
    lcfg["corectron"] = core_cfg

    # ONS (linear).
    ons_cfg = lcfg.get("ons", {"enabled": False})
    ons_radius = float(ons_cfg.get("domain_radius", 1.0))
    if ons_radius <= 0.0:
        raise ValueError("ons.domain_radius must be positive")

    h_bound_ons = 2.0 * ons_radius * g_bound_linear
    eta_fixed_ons = 1.0 / (5.0 * h_bound_ons)
    gamma_ons_theory = 1.0 / (1.0 + 2.0 * eta_fixed_ons * h_bound_ons) ** 2
    ons_lr_theory = 1.0 / gamma_ons_theory

    default_ons_lr = float(ons_cfg.get("ons_lr", 2.0))
    ons_lr = float(cli_ons_lr) if cli_ons_lr is not None else default_ons_lr
    if ons_lr <= 0.0:
        raise ValueError("ons_lr must be positive")

    gamma_ons_used = 1.0 / ons_lr
    w_diameter = 2.0 * ons_radius
    epsilon_ons_theory = d_ref / (w_diameter * w_diameter * gamma_ons_used * gamma_ons_used)

    eps_ons_raw = ons_cfg.get("epsilon")
    epsilon_ons = (
        float(cli_ons_epsilon)
        if cli_ons_epsilon is not None
        else float(epsilon_ons_theory if eps_ons_raw is None else eps_ons_raw)
    )
    if epsilon_ons <= 0.0:
        raise ValueError("ons.epsilon must be positive")

    ons_cfg["h_bound"] = h_bound_ons
    ons_cfg["g_bound"] = g_bound_linear
    ons_cfg["eta_fixed"] = eta_fixed_ons
    ons_cfg["gamma_theory"] = gamma_ons_theory
    ons_cfg["ons_lr_theory"] = ons_lr_theory
    ons_cfg["ons_lr"] = ons_lr
    ons_cfg["gamma_used"] = gamma_ons_used
    ons_cfg["epsilon_base"] = float(d_ref / (w_diameter * w_diameter))
    ons_cfg["epsilon_theory"] = epsilon_ons_theory
    ons_cfg["epsilon"] = epsilon_ons
    lcfg["ons"] = ons_cfg

    # OGD (linear).
    ogd_cfg = lcfg.get("ogd", {"enabled": False})
    ogd_radius = float(ogd_cfg.get("domain_radius", ons_radius))
    if ogd_radius <= 0.0:
        raise ValueError("ogd.domain_radius must be positive")
    d_bound = 2.0 * ogd_radius
    ogd_lr_theory = d_bound / (g_bound_linear * math.sqrt(horizon))

    step_raw = ogd_cfg.get("step0")
    step0 = (
        float(cli_ogd_step0)
        if cli_ogd_step0 is not None
        else float(ogd_lr_theory if step_raw is None else step_raw)
    )
    if step0 <= 0.0:
        raise ValueError("ogd.step0 must be positive")

    ogd_cfg["d_bound"] = d_bound
    ogd_cfg["g_bound"] = g_bound_linear
    ogd_cfg["step_base"] = d_bound / g_bound_linear
    ogd_cfg["ogd_lr_theory"] = ogd_lr_theory
    ogd_cfg["step0"] = step0
    ogd_cfg["schedule"] = str(ogd_cfg.get("schedule", "constant"))
    lcfg["ogd"] = ogd_cfg

    # Kernel CoRectron and KONS.
    rbf_cfg = lcfg.get("corectron_rbf", {"enabled": False})
    kons_cfg = lcfg.get("kons", {"enabled": False})

    kernel_cfg = dcfg.get("kernel", {})
    kernel_type = str(cli_kernel_type if cli_kernel_type is not None else kernel_cfg.get("kernel_type", "rbf")).lower()
    if kernel_type not in {"rbf", "poly"}:
        raise ValueError("dataset.kernel.kernel_type must be one of: rbf, poly")

    kernel_sigma = float(cli_kernel_sigma if cli_kernel_sigma is not None else kernel_cfg.get("sigma_true", 1.0))
    kernel_degree = selected_degree
    kernel_coef0 = float(cli_kernel_coef0 if cli_kernel_coef0 is not None else kernel_cfg.get("coef0", 0.0))
    kernel_alpha = float(cli_kernel_alpha if cli_kernel_alpha is not None else kernel_cfg.get("alpha", 1.0))
    kernel_coeff_std = float(kernel_cfg.get("coeff_std", 1.0))
    kernel_true_weight_norm = float(kernel_cfg.get("true_weight_norm", 1.0))

    if kernel_type == "rbf":
        if kernel_sigma <= 0.0:
            raise ValueError("dataset.kernel.sigma_true must be positive for rbf kernel")
    elif kernel_type == "poly":
        if kernel_degree <= 0:
            raise ValueError("dataset.kernel.degree must be positive for poly kernel")
        if kernel_alpha <= 0.0:
            raise ValueError("dataset.kernel.alpha must be positive for poly kernel")
    if kernel_coeff_std <= 0.0:
        raise ValueError("dataset.kernel.coeff_std must be positive")
    if kernel_true_weight_norm <= 0.0:
        raise ValueError("dataset.kernel.true_weight_norm must be positive")

    # Normalized-policy assumption for unified hyperparams:
    # use homogeneous polynomial k(z,z')=(z^T z')^d (alpha=1, coef0=0)
    # so the feature norm bound is ||z||^degree.
    if kernel_type == "poly":
        if abs(kernel_coef0) > 1e-12:
            raise ValueError("Normalization policy for poly kernel requires dataset.kernel.coef0=0.")
        if abs(kernel_alpha - 1.0) > 1e-12:
            raise ValueError("Normalization policy for poly kernel requires dataset.kernel.alpha=1.")

    # Persist potentially overridden kernel settings so generator and logs stay consistent.
    kernel_cfg["kernel_type"] = kernel_type
    kernel_cfg["sigma_true"] = kernel_sigma
    kernel_cfg["degree"] = kernel_degree
    kernel_cfg["coef0"] = kernel_coef0
    kernel_cfg["alpha"] = kernel_alpha
    kernel_cfg["coeff_std"] = kernel_coeff_std
    kernel_cfg["true_weight_norm"] = kernel_true_weight_norm
    # coeff_std is the coefficient sampling scale before RKHS normalization;
    # true_weight_norm sets the final norm. Retain the existing role identifier
    # for compatibility with saved configurations.
    kernel_cfg["coeff_std_role"] = "pre_norm_shape_only"
    kernel_cfg["true_weight_norm_role"] = "rkhs_norm_target"
    dcfg["kernel"] = kernel_cfg

    kappa_from_kernel = finite_float(kernel_feature_bound, "kernel feature bound", positive=True)
    kappa_raw = float(kons_cfg.get("kappa", rbf_cfg.get("kappa", kappa_from_kernel)))
    if kappa_raw <= 0.0:
        raise ValueError("kernel kappa must be positive")
    if abs(kappa_raw - kappa_from_kernel) > 1e-12:
        raise ValueError(f"kernel kappa must match the derived feature bound {kappa_from_kernel}.")
    kappa = kappa_from_kernel

    g_bound_kernel = kappa * residual_l2_max
    # CoRectron-K uses the same B^2*d_ref experimental reference as CoRectron-L.
    lambda_rbf_theory = b_payoff * b_payoff * float(d_ref)

    rbf_cfg["kernel_type"] = kernel_type
    rbf_cfg["kernel_sigma"] = kernel_sigma
    rbf_cfg["kernel_degree"] = kernel_degree
    rbf_cfg["kernel_coef0"] = kernel_coef0
    rbf_cfg["kernel_alpha"] = kernel_alpha
    rbf_cfg["kappa"] = kappa
    rbf_cfg["g_bound"] = g_bound_kernel
    rbf_cfg["b_payoff"] = b_payoff
    rbf_cfg["lambda_base"] = float(d_ref)
    rbf_cfg["lambda_reg_theory"] = lambda_rbf_theory

    lambda_rbf_raw = rbf_cfg.get("lambda_reg")
    rbf_cfg["lambda_reg"] = (
        float(cli_corectron_rbf_lambda)
        if cli_corectron_rbf_lambda is not None
        else float(lambda_rbf_theory if lambda_rbf_raw is None else lambda_rbf_raw)
    )
    if rbf_cfg["lambda_reg"] <= 0.0:
        raise ValueError("corectron_rbf.lambda_reg must be positive")
    lcfg["corectron_rbf"] = rbf_cfg

    kons_radius = float(kons_cfg.get("domain_radius", 1.0))
    if kons_radius <= 0.0:
        raise ValueError("kons.domain_radius must be positive")

    h_bound_kons = 2.0 * kons_radius * g_bound_kernel
    eta_fixed_kons = 1.0 / (5.0 * h_bound_kons)
    gamma_kons_theory = 1.0 / (1.0 + 2.0 * eta_fixed_kons * h_bound_kons) ** 2
    kons_lr_theory = 1.0 / gamma_kons_theory

    kons_lr_raw = kons_cfg.get("kons_lr")
    kons_lr = (
        float(cli_kons_lr)
        if cli_kons_lr is not None
        else float(kons_lr_theory if kons_lr_raw is None else kons_lr_raw)
    )
    if kons_lr <= 0.0:
        raise ValueError("kons.kons_lr must be positive")

    gamma_kons_used = 1.0 / kons_lr
    d_cap_linear_proxy = d_ref  # effective-dimension proxy: linear d = action_dim * context_dim
    lambda_eff = lambda_rbf_theory
    d_cap_kernel_upper = max(1, int(math.ceil(min(horizon, horizon * g_bound_kernel * g_bound_kernel / lambda_eff))))
    epsilon_kons_theory = d_cap_linear_proxy / ((2.0 * kons_radius) * (2.0 * kons_radius) * gamma_kons_used * gamma_kons_used)
    eps_kons_raw = kons_cfg.get("epsilon")
    epsilon_kons = (
        float(cli_kons_epsilon)
        if cli_kons_epsilon is not None
        else float(epsilon_kons_theory if eps_kons_raw is None else eps_kons_raw)
    )
    if epsilon_kons <= 0.0:
        raise ValueError("kons.epsilon must be positive")

    kons_cfg["kernel_type"] = kernel_type
    kons_cfg["kernel_sigma"] = kernel_sigma
    kons_cfg["kernel_degree"] = kernel_degree
    kons_cfg["kernel_coef0"] = kernel_coef0
    kons_cfg["kernel_alpha"] = kernel_alpha
    kons_cfg["kappa"] = kappa
    kons_cfg["x_diameter"] = residual_l2_max
    kons_cfg["g_bound"] = g_bound_kernel
    kons_cfg["h_bound"] = h_bound_kons
    kons_cfg["eta_fixed"] = eta_fixed_kons
    kons_cfg["gamma_theory"] = gamma_kons_theory
    kons_cfg["kons_lr_theory"] = kons_lr_theory
    kons_cfg["kons_lr"] = kons_lr
    kons_cfg["gamma_used"] = gamma_kons_used
    kons_cfg["lambda_eff"] = lambda_eff
    kons_cfg["d_cap_linear_proxy"] = d_cap_linear_proxy
    kons_cfg["d_cap_kernel_upper"] = d_cap_kernel_upper
    kons_cfg["epsilon_base"] = float(d_cap_linear_proxy / ((2.0 * kons_radius) * (2.0 * kons_radius)))
    kons_cfg["epsilon_theory"] = epsilon_kons_theory
    kons_cfg["epsilon"] = epsilon_kons

    lcfg["kons"] = kons_cfg

    if ogd_cfg["schedule"] not in {"constant", "inverse_sqrt"}:
        raise ValueError("ogd.schedule must be constant or inverse_sqrt")
    for name in ("ons", "kons"):
        lcfg[name]["proj_tol"] = finite_float(lcfg[name].get("proj_tol", 1e-10), f"{name}.proj_tol", positive=True)
    kernel_cfg["num_centers"] = integer(kernel_cfg.get("num_centers", 16), "dataset.kernel.num_centers", minimum=1)
    # These flags describe comparator feasibility, not an empirical claim of
    # a regret guarantee for misspecified models. None means not verified.
    linear_kernel = kernel_type == "poly" and kernel_degree == 1
    ons_norm = linear_true_norm if model_type == "linear_contextual" else (kernel_true_norm if linear_kernel else None)
    kons_norm = kernel_true_norm if model_type == "kernel_contextual" else (linear_true_norm if linear_kernel else None)
    ons_cfg["true_comparator_in_domain"] = None if ons_norm is None else ons_norm <= ons_radius
    kons_cfg["true_comparator_in_domain"] = None if kons_norm is None else kons_norm <= kons_radius
    cfg["learners"] = lcfg

    # User policy (suboptimal feedback).
    _VALID_SUBOPT_MODES = {"uniform", "one_swap", "score_perturb", "score_perturb_relative"}
    up_cfg = cfg.get("user_policy", {})
    if not isinstance(up_cfg, dict):
        raise ValueError("user_policy must be a mapping when provided")
    raw_subopt = up_cfg.get("subopt_prob", 0.0)
    subopt_prob = float(cli_subopt_prob) if cli_subopt_prob is not None else float(raw_subopt)
    if not (0.0 <= subopt_prob <= 1.0):
        raise ValueError(f"user_policy.subopt_prob must be in [0, 1], got {subopt_prob}")
    raw_mode = up_cfg.get("subopt_mode", "uniform")
    subopt_mode = str(cli_subopt_mode) if cli_subopt_mode is not None else str(raw_mode)
    if subopt_mode not in _VALID_SUBOPT_MODES:
        raise ValueError(f"user_policy.subopt_mode must be one of {_VALID_SUBOPT_MODES}, got {subopt_mode!r}")
    raw_noise_std = up_cfg.get("subopt_noise_std", 0.0)
    subopt_noise_std = float(cli_subopt_noise_std) if cli_subopt_noise_std is not None else float(raw_noise_std)
    if subopt_noise_std < 0.0:
        raise ValueError(f"user_policy.subopt_noise_std must be >= 0, got {subopt_noise_std}")
    up_cfg["subopt_prob"] = subopt_prob
    up_cfg["subopt_mode"] = subopt_mode
    up_cfg["subopt_noise_std"] = subopt_noise_std
    cfg["user_policy"] = up_cfg

    finite_tree(cfg)
    return cfg


def dump_resolved_config(path: Path, cfg: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def hyperparam_print_blob(cfg: dict[str, Any]) -> str:
    payload = {
        "experiment": cfg["experiment"],
        "dataset": cfg["dataset"],
        "learners": cfg["learners"],
    }
    return to_pretty_json(payload)
