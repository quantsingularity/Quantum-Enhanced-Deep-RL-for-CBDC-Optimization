"""
Run all experiments: Classical SAC, Quantum SAC, and Rule-based baseline.

Run from project root:
    PYTHONPATH=code python code/experiments/run_all_experiments.py
"""

# ── sys.path bootstrap (must come before local imports) ───────────────────────
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIGS = _ROOT / "infrastructure" / "configs"

_code_dir = str(_ROOT / "code")
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

# ── Third-party imports ───────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml

# ── Local imports (require sys.path bootstrap above) ─────────────────────────
from baselines.rule_based_policy import evaluate_rule_based
from env.cbdc_env import CBDCLiquidityEnv
from models.sac_agent import SACAgent
from training.train_qsac import evaluate as evaluate_qsac
from training.train_qsac import train_qsac
from training.train_sac import evaluate as evaluate_sac
from training.train_sac import train_sac


def run_all_experiments() -> dict:
    """Run complete experimental suite and return results dict."""

    print("=" * 80)
    print("CBDC Liquidity Management: Full Experimental Suite")
    print("=" * 80)

    log_dir = Path("logs")
    metrics_dir = log_dir / "metrics"
    plots_dir = log_dir / "plots"
    models_dir = log_dir / "trained_models"

    for d in (log_dir, metrics_dir, plots_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(str(_CONFIGS / "environment.yaml")) as f:
        env_config = yaml.safe_load(f)

    with open(str(_CONFIGS / "default.yaml")) as f:
        default_config = yaml.safe_load(f)

    seed = default_config["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results: dict = {}

    # ── 1. Classical SAC ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("1. Classical SAC")
    print("=" * 80)

    if not (models_dir / "sac_final.pt").exists():
        train_sac(
            n_steps=default_config["n_total_steps"],
            eval_freq=default_config["eval_freq"],
            n_eval_episodes=default_config["n_eval_episodes"],
        )
    else:
        print("Classical SAC model found: skipping training.")

    if (models_dir / "sac_final.pt").exists():
        env = CBDCLiquidityEnv(seed=seed, **env_config)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        sac_agent = SACAgent.load(
            str(models_dir / "sac_final.pt"),
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )
        sac_metrics = evaluate_sac(sac_agent, env, n_episodes=100)
        all_results["Classical SAC"] = sac_metrics
        env.close()

        print("Classical SAC results:")
        for k, v in sac_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("Classical SAC model not found: skipping evaluation.")
        state_dim = 8  # env defaults
        action_dim = 3

    # ── 2. Quantum SAC ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("2. Quantum-Enhanced SAC")
    print("=" * 80)

    if not (models_dir / "qsac_final.pt").exists():
        train_qsac(
            n_steps=default_config["n_total_steps"],
            eval_freq=default_config["eval_freq"],
            n_eval_episodes=default_config["n_eval_episodes"],
        )
    else:
        print("Quantum SAC model found: skipping training.")

    if (models_dir / "qsac_final.pt").exists():
        env = CBDCLiquidityEnv(seed=seed + 1, **env_config)
        # Re-derive dims in case Classical SAC block was skipped
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        with open(str(_CONFIGS / "qsac.yaml")) as f:
            qsac_config = yaml.safe_load(f)

        quantum_config = {
            "embedding_dim": qsac_config["critic_embedding_dim"],
            "n_qubits": qsac_config["n_qubits"],
            "n_vqc_layers": qsac_config["n_vqc_layers"],
            "quantum_output_dim": qsac_config["quantum_output_dim"],
            "output_dims": qsac_config["critic_output_dims"],
            "quantum_backend": qsac_config["quantum_backend"],
            "enable_zne": qsac_config["enable_zne"],
        }

        qsac_agent = SACAgent.load(
            str(models_dir / "qsac_final.pt"),
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            use_quantum_critic=True,
            quantum_config=quantum_config,
        )
        qsac_metrics = evaluate_qsac(qsac_agent, env, n_episodes=100)
        all_results["Quantum SAC"] = qsac_metrics
        env.close()

        print("Quantum SAC results:")
        for k, v in qsac_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        print("Quantum SAC model not found: skipping evaluation.")

    # ── 3. Rule-based baseline ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("3. Rule-Based Baseline")
    print("=" * 80)

    env = CBDCLiquidityEnv(seed=seed + 2, **env_config)
    rule_metrics = evaluate_rule_based(env, n_episodes=100, seed=seed + 2)
    all_results["Rule-Based"] = rule_metrics
    env.close()

    print("Rule-Based results:")
    for k, v in rule_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    if all_results:
        comparison_df = pd.DataFrame(all_results).T
        comparison_df.to_csv(metrics_dir / "all_methods_comparison.csv")
        print(f"\nResults saved to {metrics_dir / 'all_methods_comparison.csv'}")
        print("\n" + comparison_df.to_string())
        generate_comparison_plots(comparison_df, plots_dir)

    print("\n" + "=" * 80)
    print("All experiments completed.")
    print("=" * 80)
    return all_results


def generate_comparison_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    """Generate bar-chart comparisons for all metrics present in *df*."""
    sns.set_style("whitegrid")
    methods = list(df.index)
    n = len(methods)
    palette = sns.color_palette("Set2", n)

    for metric, ylabel, title, filename in [
        (
            "mean_funding_cost",
            "Mean Funding Cost ($)",
            "Funding Cost Comparison",
            "funding_cost_comparison.png",
        ),
        (
            "lcr_violation_rate",
            "LCR Violation Rate (%)",
            "LCR Compliance Comparison",
            "lcr_violation_comparison.png",
        ),
        (
            "mean_reward",
            "Mean Reward",
            "Policy Performance Comparison",
            "reward_comparison.png",
        ),
    ]:
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(max(8, n * 2), 6))
        values = df[metric] * (100 if "rate" in metric else 1)
        yerr = (
            df.get("std_" + metric.replace("mean_", ""), None)
            if metric.startswith("mean_")
            else None
        )
        bars = ax.bar(
            methods, values, yerr=yerr, capsize=5, alpha=0.8, color=palette[:n]
        )
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=300)
        print(f"Saved: {plots_dir / filename}")
        plt.close()


if __name__ == "__main__":
    run_all_experiments()
