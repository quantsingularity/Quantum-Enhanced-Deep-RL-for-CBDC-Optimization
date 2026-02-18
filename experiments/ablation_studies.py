"""
Ablation studies for quantum critic.
"""

import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.cbdc_env import CBDCLiquidityEnv
from models.sac_agent import SACAgent
from training.replay_buffer import ReplayBuffer


def run_ablation_studies():
    """
    Run ablation studies on quantum critic components.

    Studies:
        1. Baseline: Full quantum critic
        2. Reduced qubits: 4 → 2 qubits
        3. Reduced layers: 2 → 1 VQC layer
        4. No entanglement: Remove CNOT gates
        5. Classical only: Remove quantum layer entirely
    """

    print("=" * 80)
    print("Ablation Studies for Quantum Critic")
    print("=" * 80)

    # Load configs
    with open("configs/qsac.yaml", "r") as f:
        qsac_config = yaml.safe_load(f)

    with open("configs/environment.yaml", "r") as f:
        env_config = yaml.safe_load(f)

    with open("configs/default.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    seed = default_config["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create directories
    results_dir = Path("logs/ablation_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ablation configurations
    ablation_configs = {
        "Full Quantum (Baseline)": {
            "n_qubits": 4,
            "n_vqc_layers": 2,
            "entanglement_type": "ring",
            "use_quantum": True,
        },
        "Reduced Qubits (2)": {
            "n_qubits": 2,
            "n_vqc_layers": 2,
            "entanglement_type": "ring",
            "use_quantum": True,
        },
        "Reduced Layers (1)": {
            "n_qubits": 4,
            "n_vqc_layers": 1,
            "entanglement_type": "ring",
            "use_quantum": True,
        },
        "No Entanglement": {
            "n_qubits": 4,
            "n_vqc_layers": 2,
            "entanglement_type": "linear",  # Minimal entanglement
            "use_quantum": True,
        },
        "Classical Only": {
            "n_qubits": 4,
            "n_vqc_layers": 2,
            "entanglement_type": "ring",
            "use_quantum": False,
        },
    }

    # Results storage
    all_results = {}

    # Run each ablation
    for name, ablation_config in ablation_configs.items():
        print(f"\n{'='*80}")
        print(f"Running: {name}")
        print(f"{'='*80}")

        # Train agent with ablation config
        results = train_ablation(
            name=name,
            ablation_config=ablation_config,
            qsac_config=qsac_config,
            env_config=env_config,
            seed=seed,
            device=device,
            n_steps=200000,  # Reduced for ablation
        )

        all_results[name] = results

        print(f"\nResults for {name}:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(all_results).T

    # Save results
    comparison_df.to_csv(results_dir / "ablation_comparison.csv")
    print(f"\nAblation results saved to {results_dir / 'ablation_comparison.csv'}")

    print("\n" + "=" * 80)
    print("Ablation Study Summary")
    print("=" * 80)
    print(comparison_df.to_string())

    # Generate plots
    generate_ablation_plots(comparison_df, results_dir)

    return all_results


def train_ablation(
    name: str,
    ablation_config: dict,
    qsac_config: dict,
    env_config: dict,
    seed: int,
    device: str,
    n_steps: int = 200000,
):
    """Train agent with ablation configuration."""

    # Create environment
    env = CBDCLiquidityEnv(seed=seed, **env_config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create agent with ablation config
    if ablation_config["use_quantum"]:
        quantum_config = {
            "embedding_dim": qsac_config["critic_embedding_dim"],
            "n_qubits": ablation_config["n_qubits"],
            "n_vqc_layers": ablation_config["n_vqc_layers"],
            "quantum_output_dim": qsac_config["quantum_output_dim"],
            "output_dims": qsac_config["critic_output_dims"],
            "quantum_backend": qsac_config["quantum_backend"],
            "enable_zne": False,
        }

        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_hidden_dims=tuple(qsac_config["actor_hidden_dims"]),
            lr_actor=qsac_config["learning_rate_actor"],
            lr_critic=qsac_config["learning_rate_critic"],
            lr_alpha=qsac_config["learning_rate_alpha"],
            gamma=qsac_config["gamma"],
            tau=qsac_config["tau"],
            auto_entropy_tuning=qsac_config["auto_entropy_tuning"],
            device=device,
            use_quantum_critic=True,
            quantum_config=quantum_config,
        )
    else:
        # Classical critic
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_hidden_dims=tuple(qsac_config["actor_hidden_dims"]),
            critic_hidden_dims=tuple(qsac_config["critic_output_dims"]),
            lr_actor=qsac_config["learning_rate_actor"],
            lr_critic=qsac_config["learning_rate_critic"],
            lr_alpha=qsac_config["learning_rate_alpha"],
            gamma=qsac_config["gamma"],
            tau=qsac_config["tau"],
            auto_entropy_tuning=qsac_config["auto_entropy_tuning"],
            device=device,
            use_quantum_critic=False,
        )

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=qsac_config["buffer_size"],
        device=device,
    )

    # Training loop
    state, _ = env.reset()
    episode_reward = 0
    training_rewards = []

    pbar = tqdm(total=n_steps, desc=f"Training {name}")

    for step in range(n_steps):
        # Select action
        if step < qsac_config["learning_starts"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        replay_buffer.add(state, action, reward, next_state, float(terminated))

        state = next_state
        episode_reward += reward

        # Update agent
        if step >= qsac_config["learning_starts"]:
            batch = replay_buffer.sample(qsac_config["batch_size"])
            agent.update(*batch)

        # Episode end
        if done:
            training_rewards.append(episode_reward)
            state, _ = env.reset()
            episode_reward = 0

        pbar.update(1)

    pbar.close()

    # Final evaluation
    eval_metrics = evaluate_ablation(agent, env, n_episodes=50)

    env.close()

    return eval_metrics


def evaluate_ablation(agent: SACAgent, env: CBDCLiquidityEnv, n_episodes: int = 50):
    """Evaluate ablation agent."""

    episode_rewards = []
    episode_funding_costs = []
    episode_lcr_violations = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)
        episode_funding_costs.append(info["episode_funding_cost"])
        episode_lcr_violations.append(info["episode_lcr_violations"])

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_funding_cost": np.mean(episode_funding_costs),
        "lcr_violation_rate": np.mean([v > 0 for v in episode_lcr_violations]),
    }


def generate_ablation_plots(df: pd.DataFrame, results_dir: Path):
    """Generate ablation study plots."""

    sns.set_style("whitegrid")

    # Plot 1: Mean Reward
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = df.index
    rewards = df["mean_reward"]

    colors = ["green", "orange", "orange", "orange", "red"]
    bars = ax.barh(configs, rewards, color=colors, alpha=0.7)

    ax.set_xlabel("Mean Reward", fontsize=12)
    ax.set_title(
        "Ablation Study: Mean Reward Comparison", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "ablation_reward.png", dpi=300)
    print(f"Saved: {results_dir / 'ablation_reward.png'}")
    plt.close()

    # Plot 2: Funding Cost
    fig, ax = plt.subplots(figsize=(12, 6))

    costs = df["mean_funding_cost"]
    bars = ax.barh(configs, costs, color=colors, alpha=0.7)

    ax.set_xlabel("Mean Funding Cost ($)", fontsize=12)
    ax.set_title(
        "Ablation Study: Funding Cost Comparison", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "ablation_cost.png", dpi=300)
    print(f"Saved: {results_dir / 'ablation_cost.png'}")
    plt.close()

    # Plot 3: Multi-metric heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize metrics
    df_norm = df[["mean_reward", "mean_funding_cost", "lcr_violation_rate"]].copy()
    for col in df_norm.columns:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (
            df_norm[col].max() - df_norm[col].min() + 1e-8
        )

    sns.heatmap(
        df_norm,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        ax=ax,
        cbar_kws={"label": "Normalized Score"},
    )

    ax.set_title("Ablation Study: Multi-Metric Heatmap", fontsize=14, fontweight="bold")
    ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(results_dir / "ablation_heatmap.png", dpi=300)
    print(f"Saved: {results_dir / 'ablation_heatmap.png'}")
    plt.close()


if __name__ == "__main__":
    run_ablation_studies()
