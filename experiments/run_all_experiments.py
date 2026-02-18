"""
Run all experiments: Classical SAC, Quantum SAC, and Rule-based baseline.
"""

import os
import sys
import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.cbdc_env import CBDCLiquidityEnv
from training.train_sac import train_sac, evaluate as evaluate_sac
from training.train_qsac import train_qsac, evaluate as evaluate_qsac
from baselines.rule_based_policy import evaluate_rule_based
from models.sac_agent import SACAgent


def run_all_experiments():
    """Run complete experimental suite."""

    print("=" * 80)
    print("CBDC Liquidity Management - Full Experimental Suite")
    print("=" * 80)

    # Create directories
    log_dir = Path("logs")
    metrics_dir = log_dir / "metrics"
    plots_dir = log_dir / "plots"
    models_dir = log_dir / "trained_models"

    for dir_path in [log_dir, metrics_dir, plots_dir, models_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Load configs
    with open("configs/environment.yaml", "r") as f:
        env_config = yaml.safe_load(f)

    with open("configs/default.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    seed = default_config["seed"]

    # Results storage
    all_results = {}

    # 1. Train Classical SAC
    print("\n" + "=" * 80)
    print("1. Training Classical SAC")
    print("=" * 80)

    if not (models_dir / "sac_final.pt").exists():
        train_sac(
            n_steps=default_config["n_total_steps"],
            eval_freq=default_config["eval_freq"],
            n_eval_episodes=default_config["n_eval_episodes"],
        )
    else:
        print("Classical SAC model already exists. Skipping training.")

    # Evaluate Classical SAC
    print("\nEvaluating Classical SAC...")
    env = CBDCLiquidityEnv(seed=seed, **env_config)

    with open("configs/sac.yaml", "r") as f:
        yaml.safe_load(f)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"

    sac_agent = SACAgent.load(
        str(models_dir / "sac_final.pt"),
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )

    sac_metrics = evaluate_sac(sac_agent, env, n_episodes=100)
    all_results["Classical SAC"] = sac_metrics

    print("Classical SAC Results:")
    for key, value in sac_metrics.items():
        print(f"  {key}: {value:.4f}")

    env.close()

    # 2. Train Quantum SAC
    print("\n" + "=" * 80)
    print("2. Training Quantum-Enhanced SAC")
    print("=" * 80)

    if not (models_dir / "qsac_final.pt").exists():
        train_qsac(
            n_steps=default_config["n_total_steps"],
            eval_freq=default_config["eval_freq"],
            n_eval_episodes=default_config["n_eval_episodes"],
        )
    else:
        print("Quantum SAC model already exists. Skipping training.")

    # Evaluate Quantum SAC
    print("\nEvaluating Quantum SAC...")
    env = CBDCLiquidityEnv(seed=seed + 1, **env_config)

    with open("configs/qsac.yaml", "r") as f:
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

    print("Quantum SAC Results:")
    for key, value in qsac_metrics.items():
        print(f"  {key}: {value:.4f}")

    env.close()

    # 3. Evaluate Rule-Based Baseline
    print("\n" + "=" * 80)
    print("3. Evaluating Rule-Based Baseline")
    print("=" * 80)

    env = CBDCLiquidityEnv(seed=seed + 2, **env_config)
    rule_based_metrics = evaluate_rule_based(env, n_episodes=100, seed=seed + 2)
    all_results["Rule-Based"] = rule_based_metrics

    print("Rule-Based Results:")
    for key, value in rule_based_metrics.items():
        print(f"  {key}: {value:.4f}")

    env.close()

    # 4. Compare Results
    print("\n" + "=" * 80)
    print("4. Results Comparison")
    print("=" * 80)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results).T

    # Save to CSV
    comparison_df.to_csv(metrics_dir / "all_methods_comparison.csv")
    print(f"\nResults saved to {metrics_dir / 'all_methods_comparison.csv'}")

    print("\n" + comparison_df.to_string())

    # 5. Generate Plots
    print("\n" + "=" * 80)
    print("5. Generating Comparison Plots")
    print("=" * 80)

    generate_comparison_plots(comparison_df, plots_dir)

    print("\n" + "=" * 80)
    print("All experiments completed successfully!")
    print("=" * 80)

    return all_results


def generate_comparison_plots(df: pd.DataFrame, plots_dir: Path):
    """Generate comparison plots."""

    sns.set_style("whitegrid")

    # 1. Funding Cost Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = df.index
    costs = df["mean_funding_cost"]
    errors = df["std_funding_cost"]

    bars = ax.bar(methods, costs, yerr=errors, capsize=5, alpha=0.7)
    bars[0].set_color("steelblue")
    bars[1].set_color("green")
    bars[2].set_color("orange")

    ax.set_ylabel("Mean Funding Cost ($)", fontsize=12)
    ax.set_title("Funding Cost Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "funding_cost_comparison.png", dpi=300)
    print(f"Saved: {plots_dir / 'funding_cost_comparison.png'}")
    plt.close()

    # 2. LCR Violation Rate Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    violation_rates = df["lcr_violation_rate"] * 100

    bars = ax.bar(methods, violation_rates, alpha=0.7)
    bars[0].set_color("steelblue")
    bars[1].set_color("green")
    bars[2].set_color("orange")

    ax.set_ylabel("LCR Violation Rate (%)", fontsize=12)
    ax.set_title("LCR Compliance Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "lcr_violation_comparison.png", dpi=300)
    print(f"Saved: {plots_dir / 'lcr_violation_comparison.png'}")
    plt.close()

    # 3. Mean Reward Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    rewards = df["mean_reward"]
    errors = df["std_reward"]

    bars = ax.bar(methods, rewards, yerr=errors, capsize=5, alpha=0.7)
    bars[0].set_color("steelblue")
    bars[1].set_color("green")
    bars[2].set_color("orange")

    ax.set_ylabel("Mean Reward", fontsize=12)
    ax.set_title("Policy Performance Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "reward_comparison.png", dpi=300)
    print(f"Saved: {plots_dir / 'reward_comparison.png'}")
    plt.close()

    # 4. Multi-metric Comparison (Normalized)
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics_to_plot = ["mean_reward", "mean_funding_cost", "lcr_violation_rate"]

    # Normalize metrics (invert cost and violation for better visualization)
    df_norm = df[metrics_to_plot].copy()
    df_norm["mean_funding_cost"] = 1 / (df_norm["mean_funding_cost"] + 1)
    df_norm["lcr_violation_rate"] = 1 - df_norm["lcr_violation_rate"]

    # Normalize to [0, 1]
    for col in df_norm.columns:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (
            df_norm[col].max() - df_norm[col].min() + 1e-8
        )

    df_norm.plot(kind="bar", ax=ax, width=0.8)

    ax.set_ylabel("Normalized Score (Higher is Better)", fontsize=12)
    ax.set_title("Multi-Metric Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticklabels(methods, rotation=0)
    ax.legend(["Reward", "Cost Efficiency", "LCR Compliance"], loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "multi_metric_comparison.png", dpi=300)
    print(f"Saved: {plots_dir / 'multi_metric_comparison.png'}")
    plt.close()


if __name__ == "__main__":
    run_all_experiments()
