"""
Statistical tests for comparing QSAC, Classical SAC, and Rule-based methods.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.cbdc_env import CBDCLiquidityEnv
from models.sac_agent import SACAgent
from baselines.rule_based_policy import RuleBasedPolicy


def collect_episode_data(agent, env, n_episodes: int = 100, use_quantum: bool = False):
    """
    Collect episode-level data for statistical testing.

    Args:
        agent: Agent or policy
        env: Environment
        n_episodes: Number of episodes
        use_quantum: Whether agent uses quantum critic

    Returns:
        Dictionary of episode data arrays
    """
    rewards = []
    funding_costs = []
    lcr_violations = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep)
        episode_reward = 0
        done = False

        while not done:
            if isinstance(agent, RuleBasedPolicy):
                action = agent.select_action(state, env)
            else:
                action = agent.select_action(state, deterministic=True)

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)
        funding_costs.append(info["episode_funding_cost"])
        lcr_violations.append(info["episode_lcr_violations"])

    return {
        "rewards": np.array(rewards),
        "funding_costs": np.array(funding_costs),
        "lcr_violations": np.array(lcr_violations),
    }


def paired_t_test(data1: np.ndarray, data2: np.ndarray, metric_name: str):
    """
    Perform paired t-test.

    Args:
        data1: Data from method 1
        data2: Data from method 2
        metric_name: Name of metric being tested

    Returns:
        Dictionary with test results
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(data1, data2)

    # Effect size (Cohen's d)
    diff = data1 - data2
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-8)

    # Confidence interval
    ci = stats.t.interval(0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff))

    return {
        "metric": metric_name,
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "mean_diff": np.mean(diff),
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "significant": p_value < 0.05,
    }


def wilcoxon_test(data1: np.ndarray, data2: np.ndarray, metric_name: str):
    """
    Perform Wilcoxon signed-rank test (non-parametric).

    Args:
        data1: Data from method 1
        data2: Data from method 2
        metric_name: Name of metric being tested

    Returns:
        Dictionary with test results
    """
    statistic, p_value = stats.wilcoxon(data1, data2)

    return {
        "metric": metric_name,
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
):
    """
    Compute bootstrap confidence interval.

    Args:
        data: Data array
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    alpha = 1 - confidence

    ci_lower = np.percentile(means, alpha / 2 * 100)
    ci_upper = np.percentile(means, (1 - alpha / 2) * 100)

    return np.mean(data), ci_lower, ci_upper


def run_statistical_tests():
    """Run complete statistical testing suite."""

    print("=" * 80)
    print("Statistical Testing Suite")
    print("=" * 80)

    # Load configs
    with open("configs/environment.yaml", "r") as f:
        env_config = yaml.safe_load(f)

    with open("configs/default.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    seed = default_config["seed"]
    device = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"

    # Create directories
    results_dir = Path("logs/statistical_tests")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    models_dir = Path("logs/trained_models")

    # 1. Collect data from all methods
    print("\n" + "=" * 80)
    print("Collecting Episode Data")
    print("=" * 80)

    all_data = {}

    # Classical SAC
    if (models_dir / "sac_final.pt").exists():
        print("\nCollecting Classical SAC data...")
        env = CBDCLiquidityEnv(seed=seed, **env_config)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        sac_agent = SACAgent.load(
            str(models_dir / "sac_final.pt"),
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

        all_data["Classical SAC"] = collect_episode_data(sac_agent, env, n_episodes=100)
        env.close()

    # Quantum SAC
    if (models_dir / "qsac_final.pt").exists():
        print("Collecting Quantum SAC data...")
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

        all_data["Quantum SAC"] = collect_episode_data(
            qsac_agent, env, n_episodes=100, use_quantum=True
        )
        env.close()

    # Rule-based
    print("Collecting Rule-Based data...")
    env = CBDCLiquidityEnv(seed=seed + 2, **env_config)
    rule_based_policy = RuleBasedPolicy()
    all_data["Rule-Based"] = collect_episode_data(
        rule_based_policy, env, n_episodes=100
    )
    env.close()

    # 2. Paired t-tests
    print("\n" + "=" * 80)
    print("Paired T-Tests")
    print("=" * 80)

    t_test_results = []

    # QSAC vs Classical SAC
    if "Quantum SAC" in all_data and "Classical SAC" in all_data:
        print("\nQuantum SAC vs Classical SAC:")

        for metric in ["rewards", "funding_costs", "lcr_violations"]:
            result = paired_t_test(
                all_data["Quantum SAC"][metric],
                all_data["Classical SAC"][metric],
                metric,
            )
            t_test_results.append({"comparison": "QSAC vs Classical SAC", **result})

            print(f"\n  {metric}:")
            print(f"    Mean diff: {result['mean_diff']:.4f}")
            print(f"    t-statistic: {result['t_statistic']:.4f}")
            print(f"    p-value: {result['p_value']:.4e}")
            print(f"    Cohen's d: {result['cohens_d']:.4f}")
            print(f"    Significant: {result['significant']}")

    # QSAC vs Rule-based
    if "Quantum SAC" in all_data and "Rule-Based" in all_data:
        print("\n\nQuantum SAC vs Rule-Based:")

        for metric in ["rewards", "funding_costs", "lcr_violations"]:
            result = paired_t_test(
                all_data["Quantum SAC"][metric], all_data["Rule-Based"][metric], metric
            )
            t_test_results.append({"comparison": "QSAC vs Rule-Based", **result})

            print(f"\n  {metric}:")
            print(f"    Mean diff: {result['mean_diff']:.4f}")
            print(f"    t-statistic: {result['t_statistic']:.4f}")
            print(f"    p-value: {result['p_value']:.4e}")
            print(f"    Cohen's d: {result['cohens_d']:.4f}")
            print(f"    Significant: {result['significant']}")

    # Save t-test results
    t_test_df = pd.DataFrame(t_test_results)
    t_test_df.to_csv(results_dir / "paired_t_tests.csv", index=False)
    print(f"\n\nT-test results saved to {results_dir / 'paired_t_tests.csv'}")

    # 3. Wilcoxon signed-rank tests
    print("\n" + "=" * 80)
    print("Wilcoxon Signed-Rank Tests")
    print("=" * 80)

    wilcoxon_results = []

    # QSAC vs Classical SAC
    if "Quantum SAC" in all_data and "Classical SAC" in all_data:
        print("\nQuantum SAC vs Classical SAC:")

        for metric in ["rewards", "funding_costs", "lcr_violations"]:
            result = wilcoxon_test(
                all_data["Quantum SAC"][metric],
                all_data["Classical SAC"][metric],
                metric,
            )
            wilcoxon_results.append({"comparison": "QSAC vs Classical SAC", **result})

            print(
                f"  {metric}: p-value={result['p_value']:.4e}, significant={result['significant']}"
            )

    # Save Wilcoxon results
    wilcoxon_df = pd.DataFrame(wilcoxon_results)
    wilcoxon_df.to_csv(results_dir / "wilcoxon_tests.csv", index=False)
    print(f"\nWilcoxon results saved to {results_dir / 'wilcoxon_tests.csv'}")

    # 4. Bootstrap confidence intervals
    print("\n" + "=" * 80)
    print("Bootstrap Confidence Intervals")
    print("=" * 80)

    bootstrap_results = []

    for method_name, data in all_data.items():
        print(f"\n{method_name}:")

        for metric in ["rewards", "funding_costs"]:
            mean, ci_lower, ci_upper = bootstrap_confidence_interval(
                data[metric], n_bootstrap=10000
            )

            bootstrap_results.append(
                {
                    "method": method_name,
                    "metric": metric,
                    "mean": mean,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "ci_width": ci_upper - ci_lower,
                }
            )

            print(f"  {metric}: {mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Save bootstrap results
    bootstrap_df = pd.DataFrame(bootstrap_results)
    bootstrap_df.to_csv(results_dir / "bootstrap_ci.csv", index=False)
    print(f"\nBootstrap CI saved to {results_dir / 'bootstrap_ci.csv'}")

    # 5. Generate plots
    print("\n" + "=" * 80)
    print("Generating Statistical Plots")
    print("=" * 80)

    generate_statistical_plots(all_data, bootstrap_df, results_dir)

    print("\n" + "=" * 80)
    print("Statistical testing complete!")
    print("=" * 80)


def generate_statistical_plots(
    all_data: Dict, bootstrap_df: pd.DataFrame, results_dir: Path
):
    """Generate statistical comparison plots."""

    sns.set_style("whitegrid")

    # Plot 1: Distribution comparison (funding costs)
    fig, ax = plt.subplots(figsize=(12, 6))

    for method_name, data in all_data.items():
        ax.hist(
            data["funding_costs"], alpha=0.5, bins=30, label=method_name, density=True
        )

    ax.set_xlabel("Funding Cost ($)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Funding Cost Distribution Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "funding_cost_distribution.png", dpi=300)
    print(f"Saved: {results_dir / 'funding_cost_distribution.png'}")
    plt.close()

    # Plot 2: Bootstrap CI visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    funding_cost_df = bootstrap_df[bootstrap_df["metric"] == "funding_costs"]

    methods = funding_cost_df["method"]
    means = funding_cost_df["mean"]
    ci_lowers = funding_cost_df["ci_lower"]
    ci_uppers = funding_cost_df["ci_upper"]

    y_pos = np.arange(len(methods))

    ax.errorbar(
        means,
        y_pos,
        xerr=[means - ci_lowers, ci_uppers - means],
        fmt="o",
        markersize=8,
        capsize=5,
        capthick=2,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Funding Cost ($)", fontsize=12)
    ax.set_title(
        "Funding Cost with 95% Bootstrap Confidence Intervals",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "bootstrap_ci_plot.png", dpi=300)
    print(f"Saved: {results_dir / 'bootstrap_ci_plot.png'}")
    plt.close()


if __name__ == "__main__":
    run_statistical_tests()
