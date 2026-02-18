"""
Evaluate trained models.
"""

import yaml
import torch
import numpy as np
from pathlib import Path
import pandas as pd

from env.cbdc_env import CBDCLiquidityEnv
from models.sac_agent import SACAgent


def evaluate_model(
    model_path: str,
    env_config_path: str = "configs/environment.yaml",
    n_episodes: int = 100,
    seed: int = 42,
    use_quantum: bool = False,
):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to model checkpoint
        env_config_path: Path to environment config
        n_episodes: Number of evaluation episodes
        seed: Random seed
        use_quantum: Whether model uses quantum critic

    Returns:
        Dictionary of evaluation metrics and episode data
    """
    # Load environment config
    with open(env_config_path, "r") as f:
        env_config = yaml.safe_load(f)

    # Create environment
    env = CBDCLiquidityEnv(seed=seed, **env_config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load agent
    if use_quantum:
        # Need to specify quantum config
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

        agent = SACAgent.load(
            model_path,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            use_quantum_critic=True,
            quantum_config=quantum_config,
        )
    else:
        agent = SACAgent.load(
            model_path,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            use_quantum_critic=False,
        )

    # Evaluate
    episode_rewards = []
    episode_funding_costs = []
    episode_lcr_violations = []
    episode_liquidity_histories = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        episode_reward = 0
        liquidity_history = []
        done = False

        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            liquidity_history.append(info["liquidity"])

        episode_rewards.append(episode_reward)
        episode_funding_costs.append(info["episode_funding_cost"])
        episode_lcr_violations.append(info["episode_lcr_violations"])
        episode_liquidity_histories.append(liquidity_history)

    # Compute metrics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "median_reward": np.median(episode_rewards),
        "mean_funding_cost": np.mean(episode_funding_costs),
        "std_funding_cost": np.std(episode_funding_costs),
        "median_funding_cost": np.median(episode_funding_costs),
        "lcr_violation_rate": np.mean([v > 0 for v in episode_lcr_violations]),
        "mean_lcr_violations": np.mean(episode_lcr_violations),
        "total_episodes": n_episodes,
    }

    # Liquidity buffer statistics
    all_liquidity = np.concatenate(episode_liquidity_histories)
    metrics["liquidity_mean"] = np.mean(all_liquidity)
    metrics["liquidity_std"] = np.std(all_liquidity)
    metrics["liquidity_min"] = np.min(all_liquidity)
    metrics["liquidity_sharpe"] = np.mean(all_liquidity) / (
        np.std(all_liquidity) + 1e-8
    )

    # Drawdown
    max_drawdown = 0
    for history in episode_liquidity_histories:
        running_max = np.maximum.accumulate(history)
        drawdown = (running_max - history) / (running_max + 1e-8)
        max_drawdown = max(max_drawdown, np.max(drawdown))
    metrics["max_liquidity_drawdown"] = max_drawdown

    env.close()

    return metrics, {
        "rewards": episode_rewards,
        "funding_costs": episode_funding_costs,
        "lcr_violations": episode_lcr_violations,
        "liquidity_histories": episode_liquidity_histories,
    }


def main():
    """Evaluate all trained models and generate comparison."""

    models_dir = Path("logs/trained_models")
    results_dir = Path("logs/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Models to evaluate
    models = {
        "Classical SAC": {
            "path": models_dir / "sac_final.pt",
            "use_quantum": False,
        },
        "Quantum SAC": {
            "path": models_dir / "qsac_final.pt",
            "use_quantum": True,
        },
    }

    # Evaluate each model
    all_results = {}

    for name, config in models.items():
        if not config["path"].exists():
            print(f"Model not found: {config['path']}")
            continue

        print(f"\nEvaluating {name}...")
        metrics, episode_data = evaluate_model(
            str(config["path"]),
            use_quantum=config["use_quantum"],
            n_episodes=100,
        )

        all_results[name] = {"metrics": metrics, "episode_data": episode_data}

        print(f"Results for {name}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(
        {name: results["metrics"] for name, results in all_results.items()}
    ).T

    # Save to CSV
    comparison_df.to_csv(results_dir / "model_comparison.csv")
    print(f"\nComparison saved to {results_dir / 'model_comparison.csv'}")

    # Print comparison
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(comparison_df.to_string())

    return all_results


if __name__ == "__main__":
    main()
