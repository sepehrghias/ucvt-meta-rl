"""Sequential (multi-step) proof of concept for UC-VT.

This extends the one-step contextual-bandit PoC (ucvt_poc.py) to a genuine
temporal-credit-assignment setting: a short-horizon goal-reaching task where
the reward at each step depends on an unknown, task-specific goal position,
and the policy must plan several steps ahead (large single-step corrections
are penalized by an action cost, so the optimal behavior is to approach the
goal gradually).

Everything stays CPU-only and closed-form:
- The environment has KNOWN, task-independent deterministic dynamics
  (s' = s + step_size * a, clipped to bounds).
- The task-specific unknown is the REWARD function: r(s, a, z) depends on a
  hidden goal position g(z). Agents only ever observe noisy rewards from
  interaction; g(z) is never given to the learner.
- Real-task and virtual-task Q-functions are fit with Fitted-Q Iteration
  (FQI): a few rounds of ridge regression on polynomial features of
  (s, a, z), with Bellman targets recomputed each round via a grid search
  over actions on the next state.
- Virtual (synthetic) transitions come from a bootstrap ensemble of reward
  decoders trained on real interaction data; ensemble disagreement is the
  uncertainty signal, exactly as in the one-step PoC.

No neural networks, no gradient descent, no GPU. Designed to run in a few
minutes on a laptop.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


METHODS = (
    "real_only",
    "unweighted",
    "uniform_matched",
    "random_weighted",
    "hard_rejection",
    "uncertainty_weighted",
    "oracle_weighted",
)

DISPLAY_NAMES = {
    "real_only": "Real only",
    "unweighted": "Unweighted VT",
    "uniform_matched": "Uniform matched",
    "random_weighted": "Random weights",
    "hard_rejection": "Hard rejection",
    "uncertainty_weighted": "UC-VT",
    "oracle_weighted": "Oracle diagnostic",
}


@dataclass(frozen=True)
class Config:
    seeds: int = 25
    train_limit: float = 0.6
    virtual_limit: float = 1.8
    n_train_tasks: int = 10
    n_virtual_tasks: int = 41

    horizon: int = 6
    gamma: float = 0.90
    step_size: float = 0.5
    state_low: float = -2.5
    state_high: float = 2.5
    action_low: float = -2.0
    action_high: float = 2.0
    action_penalty: float = 0.05
    reward_noise: float = 0.03

    episodes_per_task: int = 8          # random-behavior rollouts per real task
    state_grid_size: int = 11           # support grid for virtual transitions
    action_grid_size_support: int = 15  # action support grid for virtual transitions
    action_grid_size_eval: int = 41     # action grid used for greedy eval / FQI max

    feature_degree: int = 3             # joint poly degree over (s, a, z)
    reward_ridge: float = 1e-3
    q_ridge: float = 5e-3
    ensemble_size: int = 9
    fqi_iterations: int = 12

    virtual_strength: float = 0.5
    uncertainty_beta: float = 16.0
    uncertainty_scale_quantile: float = 0.65

    @property
    def ood_bins(self) -> tuple[tuple[str, float, float], ...]:
        return (
            ("near", 0.70, 0.95),
            ("mid", 0.95, 1.20),
            ("far", 1.20, 1.50),
        )


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def goal_position(z: np.ndarray) -> np.ndarray:
    """Hidden, task-specific goal. Never revealed to the learner directly."""
    return np.sin(1.3 * z) + 0.15 * z


def step_dynamics(s: np.ndarray, a: np.ndarray, cfg: Config) -> np.ndarray:
    """Known, task-independent deterministic dynamics."""
    return np.clip(s + cfg.step_size * a, cfg.state_low, cfg.state_high)


def true_reward(s: np.ndarray, a: np.ndarray, z: np.ndarray, cfg: Config) -> np.ndarray:
    g = goal_position(z)
    return -np.square(s - g) - cfg.action_penalty * np.square(a)


# --------------------------------------------------------------------------
# Features / regression
# --------------------------------------------------------------------------

def polynomial_exponents_3d(degree: int) -> list[tuple[int, int, int]]:
    exps = []
    for total in range(degree + 1):
        for i in range(total + 1):
            for j in range(total - i + 1):
                k = total - i - j
                exps.append((i, j, k))
    return exps


_EXPONENTS_CACHE: dict[int, list[tuple[int, int, int]]] = {}


def sat_features(s: np.ndarray, a: np.ndarray, z: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial features over (state, action, task)."""
    if degree not in _EXPONENTS_CACHE:
        _EXPONENTS_CACHE[degree] = polynomial_exponents_3d(degree)
    exps = _EXPONENTS_CACHE[degree]
    s = np.asarray(s, dtype=float)
    a = np.asarray(a, dtype=float)
    z = np.asarray(z, dtype=float)
    cols = [np.power(s, i) * np.power(a, j) * np.power(z, k) for (i, j, k) in exps]
    return np.column_stack(cols)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        xtx = x.T @ x
        xty = x.T @ y
    else:
        w = np.asarray(weights, dtype=float)
        xtx = x.T @ (w[:, None] * x)
        xty = x.T @ (w * y)
    reg = alpha * np.eye(x.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(xtx + reg, xty)


# --------------------------------------------------------------------------
# Data collection (real tasks only; behavior = random exploration)
# --------------------------------------------------------------------------

def collect_real_dataset(rng: np.random.Generator, z_train: np.ndarray, cfg: Config):
    s_list, a_list, r_list, sp_list, z_list = [], [], [], [], []
    for z in z_train:
        for _ in range(cfg.episodes_per_task):
            s = 0.0
            for _ in range(cfg.horizon):
                a = rng.uniform(cfg.action_low, cfg.action_high)
                r = true_reward(s, a, z, cfg) + rng.normal(0.0, cfg.reward_noise)
                sp = float(step_dynamics(np.array(s), np.array(a), cfg))
                s_list.append(s)
                a_list.append(a)
                r_list.append(r)
                sp_list.append(sp)
                z_list.append(z)
                s = sp
    return (
        np.array(s_list), np.array(a_list), np.array(r_list),
        np.array(sp_list), np.array(z_list),
    )


# --------------------------------------------------------------------------
# Virtual-task generation via bootstrap reward-decoder ensemble
# --------------------------------------------------------------------------

def fit_reward_ensemble(rng: np.random.Generator, s, a, z, r, cfg: Config) -> np.ndarray:
    x = sat_features(s, a, z, cfg.feature_degree)
    n = r.size
    coefs = []
    for _ in range(cfg.ensemble_size):
        idx = rng.integers(0, n, n)
        coefs.append(ridge_fit(x[idx], r[idx], cfg.reward_ridge))
    return np.column_stack(coefs)  # [dim, ensemble]


def build_virtual_support(z_virtual: np.ndarray, cfg: Config):
    s_support = np.linspace(cfg.state_low, cfg.state_high, cfg.state_grid_size)
    a_support = np.linspace(cfg.action_low, cfg.action_high, cfg.action_grid_size_support)
    ss, aa, zz = np.meshgrid(s_support, a_support, z_virtual, indexing="ij")
    return ss.ravel(), aa.ravel(), zz.ravel()


def virtual_transitions(ensemble_coef: np.ndarray, s_v, a_v, z_v, cfg: Config):
    x = sat_features(s_v, a_v, z_v, cfg.feature_degree)
    preds = x @ ensemble_coef  # [n, ensemble]
    r_hat = preds.mean(axis=1)
    uncertainty = preds.var(axis=1)
    sp_v = step_dynamics(s_v, a_v, cfg)  # dynamics are known -> no dynamics uncertainty
    return r_hat, sp_v, uncertainty


def confidence_from_uncertainty(u: np.ndarray, cfg: Config) -> np.ndarray:
    scale = np.quantile(u, cfg.uncertainty_scale_quantile) + 1e-10
    return np.exp(-cfg.uncertainty_beta * u / scale)


def rank_mass_weights(score: np.ndarray, total_mass: float, lower_is_better: bool = True) -> np.ndarray:
    order = np.argsort(score if lower_is_better else -score)
    w = np.zeros_like(score, dtype=float)
    ranks = np.arange(1, score.size + 1)[::-1]  # best rank gets highest raw weight
    raw = np.zeros_like(score, dtype=float)
    raw[order] = ranks
    w = raw / raw.sum() * total_mass * score.size
    return w


# --------------------------------------------------------------------------
# Fitted-Q Iteration
# --------------------------------------------------------------------------

def greedy_max_q(theta: np.ndarray, sp: np.ndarray, z: np.ndarray, action_grid: np.ndarray, cfg: Config) -> np.ndarray:
    n = sp.size
    m = action_grid.size
    s_rep = np.repeat(sp, m)
    z_rep = np.repeat(z, m)
    a_rep = np.tile(action_grid, n)
    feats = sat_features(s_rep, a_rep, z_rep, cfg.feature_degree)
    q_vals = (feats @ theta).reshape(n, m)
    return q_vals.max(axis=1)


def fitted_q_iteration(
    s_real, a_real, r_real, sp_real, z_real,
    s_virt, a_virt, r_virt, sp_virt, z_virt, w_virt,
    cfg: Config,
) -> np.ndarray:
    action_grid = np.linspace(cfg.action_low, cfg.action_high, cfg.action_grid_size_eval)

    s_all = np.concatenate([s_real, s_virt])
    a_all = np.concatenate([a_real, a_virt])
    r_all = np.concatenate([r_real, r_virt])
    sp_all = np.concatenate([sp_real, sp_virt])
    z_all = np.concatenate([z_real, z_virt])
    w_all = np.concatenate([np.ones_like(r_real), cfg.virtual_strength * w_virt])

    x_all = sat_features(s_all, a_all, z_all, cfg.feature_degree)
    theta = ridge_fit(x_all, r_all, cfg.q_ridge, weights=w_all)  # bootstrap: ignore-next-state init

    for _ in range(cfg.fqi_iterations):
        max_next_q = greedy_max_q(theta, sp_all, z_all, action_grid, cfg)
        targets = r_all + cfg.gamma * max_next_q
        theta = ridge_fit(x_all, targets, cfg.q_ridge, weights=w_all)
    return theta


def rollout_return(theta: np.ndarray, z: np.ndarray, cfg: Config) -> np.ndarray:
    """Greedy-policy rollout, evaluated with TRUE reward (evaluator only)."""
    action_grid = np.linspace(cfg.action_low, cfg.action_high, cfg.action_grid_size_eval)
    n = z.size
    s = np.zeros(n)
    total = np.zeros(n)
    disc = 1.0
    for _ in range(cfg.horizon):
        s_rep = np.repeat(s, action_grid.size)
        z_rep = np.repeat(z, action_grid.size)
        a_rep = np.tile(action_grid, n)
        feats = sat_features(s_rep, a_rep, z_rep, cfg.feature_degree)
        q_vals = (feats @ theta).reshape(n, action_grid.size)
        best_idx = np.argmax(q_vals, axis=1)
        a = action_grid[best_idx]
        r = true_reward(s, a, z, cfg)
        total += disc * r
        s = step_dynamics(s, a, cfg)
        disc *= cfg.gamma
    return total


# --------------------------------------------------------------------------
# One seed, all methods
# --------------------------------------------------------------------------

def run_seed(seed: int, z_train: np.ndarray, z_virtual: np.ndarray, z_tests: dict, cfg: Config) -> dict:
    rng = np.random.default_rng(seed)

    s_r, a_r, r_r, sp_r, z_r = collect_real_dataset(rng, z_train, cfg)

    ensemble_coef = fit_reward_ensemble(rng, s_r, a_r, z_r, r_r, cfg)
    s_v, a_v, z_v = build_virtual_support(z_virtual, cfg)
    r_hat, sp_v, uncertainty = virtual_transitions(ensemble_coef, s_v, a_v, z_v, cfg)
    r_true_v = true_reward(s_v, a_v, z_v, cfg)  # evaluator-only, for oracle diagnostic + calibration
    true_error = np.abs(r_hat - r_true_v)

    pearson = stats.pearsonr(uncertainty, true_error)[0]
    spearman = stats.spearmanr(uncertainty, true_error)[0]

    conf = confidence_from_uncertainty(uncertainty, cfg)
    hard_mask = uncertainty <= np.quantile(uncertainty, cfg.uncertainty_scale_quantile)
    total_mass = conf.sum()

    weight_sets = {
        "real_only": np.zeros_like(conf),
        "unweighted": np.ones_like(conf),
        "uniform_matched": np.full_like(conf, total_mass / conf.size),
        "random_weighted": rng.permutation(conf),
        "hard_rejection": hard_mask.astype(float),
        "uncertainty_weighted": conf,
        "oracle_weighted": rank_mass_weights(true_error, total_mass / conf.size, lower_is_better=True),
    }

    results = {}
    for method, w_virt in weight_sets.items():
        theta = fitted_q_iteration(s_r, a_r, r_r, sp_r, z_r, s_v, a_v, r_hat, sp_v, z_v, w_virt, cfg)
        row = {}
        all_returns = []
        for name, z_bin in z_tests.items():
            if name == "all":
                continue
            ret = rollout_return(theta, z_bin, cfg)
            row[name] = ret
            all_returns.append(ret)
        row["all"] = np.concatenate(all_returns)
        results[method] = row

    return {"results": results, "pearson": pearson, "spearman": spearman}


# --------------------------------------------------------------------------
# Aggregation, stats, output
# --------------------------------------------------------------------------

def mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(x.size) if x.size > 1 else 0.0
    return mean, std, ci95


def paired_comparison(a: np.ndarray, b: np.ndarray) -> dict:
    diff = a - b
    mean, std, ci95 = mean_ci(diff)
    t_stat, p_value = stats.ttest_rel(a, b) if diff.size > 1 else (float("nan"), float("nan"))
    return {
        "mean_difference": mean,
        "ci_low": mean - ci95,
        "ci_high": mean + ci95,
        "p_value": float(p_value),
        "win_rate": float(np.mean(diff > 0)),
    }


def run_main(cfg: Config, output_dir: Path):
    z_train = np.linspace(-cfg.train_limit, cfg.train_limit, cfg.n_train_tasks)
    z_virtual = np.linspace(-cfg.virtual_limit, cfg.virtual_limit, cfg.n_virtual_tasks)
    z_tests = {}
    for name, low, high in cfg.ood_bins:
        z_tests[name] = np.concatenate([np.linspace(-high, -low, 25), np.linspace(low, high, 25)])

    per_seed = {m: {"near": [], "mid": [], "far": [], "all": []} for m in METHODS}
    pearsons, spearmans = [], []

    for seed in range(cfg.seeds):
        out = run_seed(seed, z_train, z_virtual, z_tests, cfg)
        pearsons.append(out["pearson"])
        spearmans.append(out["spearman"])
        for m in METHODS:
            for split in ("near", "mid", "far", "all"):
                per_seed[m][split].append(float(np.mean(out["results"][m][split])))

    methods_agg = {}
    for m in METHODS:
        methods_agg[m] = {}
        for split in ("near", "mid", "far", "all"):
            arr = np.array(per_seed[m][split])
            mean, std, ci95 = mean_ci(arr)
            methods_agg[m][split] = {"mean": mean, "std": std, "ci95": ci95, "raw": arr.tolist()}

    comparisons = {
        "uc_vs_unweighted": paired_comparison(
            np.array(per_seed["uncertainty_weighted"]["all"]), np.array(per_seed["unweighted"]["all"])
        ),
        "uc_vs_uniform_matched": paired_comparison(
            np.array(per_seed["uncertainty_weighted"]["all"]), np.array(per_seed["uniform_matched"]["all"])
        ),
        "uc_vs_random": paired_comparison(
            np.array(per_seed["uncertainty_weighted"]["all"]), np.array(per_seed["random_weighted"]["all"])
        ),
        "uc_vs_hard": paired_comparison(
            np.array(per_seed["uncertainty_weighted"]["all"]), np.array(per_seed["hard_rejection"]["all"])
        ),
        "uc_vs_real": paired_comparison(
            np.array(per_seed["uncertainty_weighted"]["all"]), np.array(per_seed["real_only"]["all"])
        ),
    }

    aggregate = {
        "methods": methods_agg,
        "pearson_mean": float(np.mean(pearsons)),
        "pearson_std": float(np.std(pearsons, ddof=1)) if len(pearsons) > 1 else 0.0,
        "spearman_mean": float(np.mean(spearmans)),
        "spearman_std": float(np.std(spearmans, ddof=1)) if len(spearmans) > 1 else 0.0,
        "comparisons": comparisons,
    }
    return aggregate


def save_figure(output_dir: Path, aggregate: dict, cfg: Config):
    methods_for_bar = (
        "real_only", "unweighted", "random_weighted", "hard_rejection",
        "uncertainty_weighted", "oracle_weighted",
    )
    labels = [DISPLAY_NAMES[m] for m in methods_for_bar]
    means = [aggregate["methods"][m]["all"]["mean"] for m in methods_for_bar]
    cis = [aggregate["methods"][m]["all"]["ci95"] for m in methods_for_bar]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(labels))
    axes[0].bar(x, means, yerr=cis, capsize=3)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Mean discounted OOD return")
    axes[0].set_title(f"Sequential PoC: {cfg.seeds}-seed comparison (95% CI)")
    axes[0].grid(axis="y", alpha=0.2)

    distance_labels = ["Near", "Mid", "Far"]
    for method in ("real_only", "unweighted", "hard_rejection", "uncertainty_weighted"):
        axes[1].plot(
            distance_labels,
            [aggregate["methods"][method][s]["mean"] for s in ("near", "mid", "far")],
            marker="o",
            label=DISPLAY_NAMES[method],
        )
    axes[1].set_ylabel("Mean discounted return")
    axes[1].set_title("Return vs OOD distance")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "ucvt_sequential_poc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_comparison(name: str, c: dict) -> str:
    return (
        f"{name}: mean diff={c['mean_difference']:.4f}, "
        f"95% CI=[{c['ci_low']:.4f}, {c['ci_high']:.4f}], "
        f"paired p={c['p_value']:.3g}, win rate={100 * c['win_rate']:.1f}%"
    )


def save_summary(path: Path, aggregate: dict, cfg: Config):
    lines = [
        "UC-VT sequential (multi-step, Fitted-Q Iteration) proof of concept",
        f"Seeds: {cfg.seeds}, horizon: {cfg.horizon}, gamma: {cfg.gamma}",
        "Dynamics are known/deterministic; only the reward (goal position) is task-unknown.",
        "",
        "Main discounted OOD returns (mean +/- sample std):",
    ]
    for m in METHODS:
        item = aggregate["methods"][m]["all"]
        lines.append(f"  {DISPLAY_NAMES[m]}: {item['mean']:.4f} +/- {item['std']:.4f}")
    lines += [
        "",
        f"Per-seed Pearson uncertainty/error correlation: {aggregate['pearson_mean']:.3f} +/- {aggregate['pearson_std']:.3f}",
        f"Per-seed Spearman uncertainty/error correlation: {aggregate['spearman_mean']:.3f} +/- {aggregate['spearman_std']:.3f}",
        "",
        "Paired comparisons (UC-VT minus baseline):",
    ]
    for key, label in (
        ("uc_vs_unweighted", "UC-VT vs unweighted"),
        ("uc_vs_uniform_matched", "UC-VT vs uniform matched-mass"),
        ("uc_vs_random", "UC-VT vs random weighting"),
        ("uc_vs_hard", "UC-VT vs hard rejection"),
        ("uc_vs_real", "UC-VT vs real-only"),
    ):
        lines.append("  " + format_comparison(label, aggregate["comparisons"][key]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=Path("poc_seq_output"))
    args = parser.parse_args()

    cfg = Config(seeds=args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = run_main(cfg, args.output_dir)
    save_summary(args.output_dir / "summary.txt", aggregate, cfg)
    save_figure(args.output_dir, aggregate, cfg)

    def strip_raw(agg):
        return {
            "methods": {
                m: {s: {k: v for k, v in d.items() if k != "raw"} for s, d in splits.items()}
                for m, splits in agg["methods"].items()
            },
            "pearson_mean": agg["pearson_mean"],
            "pearson_std": agg["pearson_std"],
            "spearman_mean": agg["spearman_mean"],
            "spearman_std": agg["spearman_std"],
            "comparisons": agg["comparisons"],
        }

    (args.output_dir / "results.json").write_text(
        json.dumps({"config": asdict(cfg), "aggregate": strip_raw(aggregate)}, indent=2),
        encoding="utf-8",
    )
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
