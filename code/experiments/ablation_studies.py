"""
Ablation studies for quantum critic components.

Run from project root:
    PYTHONPATH=code python code/experiments/ablation_studies.py
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
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

# ── Local imports ─────────────────────────────────────────────────────────────
from env.cbdc_env import CBDCLiquidityEnv
from models.sac_agent import SACAgent
from tqdm import tqdm
from training.replay_buffer import ReplayBuffer


def run_ablation_studies() -> dict:
    """
    Run ablation studies on quantum critic components.

    Studies
    -------
    1. Full Quantum (Baseline) : 4 qubits, 2 layers, ring entanglement
    2. Reduced Qubits          : 2 qubits, 2 layers, ring entanglement
    3. Reduced Layers          : 4 qubits, 1 layer,  ring entanglement
    4. No Entanglement         : 4 qubits, 2 layers, linear (no CNOT loops)
    5. Classical Only          : standard MLP critic, no quantum layer
    """
    print("=" * 80)
    print("Ablation Studies: Quantum Critic")
    print("=" * 80)

    with open(str(_CONFIGS / "qsac.yaml")) as f:
        qsac_config = yaml.safe_load(f)
    with open(str(_CONFIGS / "environment.yaml")) as f:
        env_config = yaml.safe_load(f)
    with open(str(_CONFIGS / "default.yaml")) as f:
        default_config = yaml.safe_load(f)

    seed = default_config["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_dir = Path("logs/ablation_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Each entry: keys consumed by train_ablation()
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
        "No Entanglement (linear)": {
            "n_qubits": 4,
            "n_vqc_layers": 2,
            "entanglement_type": "linear",
            "use_quantum": True,
        },
        "Classical Only": {
            "n_qubits": 4,
            "n_vqc_layers": 2,
            "entanglement_type": "ring",
            "use_quantum": False,
        },
    }

    all_results: dict = {}

    for name, ablation_cfg in ablation_configs.items():
        print(f"\n{'=' * 80}\nRunning: {name}\n{'=' * 80}")
        results = train_ablation(
            name=name,
            ablation_config=ablation_cfg,
            qsac_config=qsac_config,
            env_config=env_config,
            seed=seed,
            device=device,
            n_steps=200_000,
        )
        all_results[name] = results
        print(f"\nResults for {name}:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    comparison_df = pd.DataFrame(all_results).T
    comparison_df.to_csv(results_dir / "ablation_comparison.csv")
    print(f"\nAblation results → {results_dir / 'ablation_comparison.csv'}")
    print("\n" + "=" * 80 + "\nAblation Summary\n" + "=" * 80)
    print(comparison_df.to_string())

    generate_ablation_plots(comparison_df, results_dir)
    return all_results


def train_ablation(
    name: str,
    ablation_config: dict,
    qsac_config: dict,
    env_config: dict,
    seed: int,
    device: str,
    n_steps: int = 200_000,
) -> dict:
    """Train one ablation variant and return evaluation metrics."""
    env = CBDCLiquidityEnv(seed=seed, **env_config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if ablation_config["use_quantum"]:
        quantum_config = {
            "embedding_dim": qsac_config["critic_embedding_dim"],
            "n_qubits": ablation_config["n_qubits"],
            "n_vqc_layers": ablation_config["n_vqc_layers"],
            "quantum_output_dim": qsac_config["quantum_output_dim"],
            "output_dims": qsac_config["critic_output_dims"],
            "quantum_backend": qsac_config["quantum_backend"],
            # Wire entanglement_type through so ablation actually changes the circuit
            "entanglement_type": ablation_config["entanglement_type"],
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

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=qsac_config["buffer_size"],
        device=device,
    )

    state, _ = env.reset()
    episode_reward = 0.0

    for step in tqdm(range(n_steps), desc=f"Training {name}", leave=False):
        if step < qsac_config["learning_starts"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.add(state, action, reward, next_state, float(terminated))
        state = next_state
        episode_reward += reward

        if step >= qsac_config["learning_starts"]:
            batch = replay_buffer.sample(qsac_config["batch_size"])
            agent.update(*batch)

        if done:
            state, _ = env.reset()
            episode_reward = 0.0

    eval_metrics = evaluate_ablation(agent, env, n_episodes=50)
    env.close()
    return eval_metrics


def evaluate_ablation(
    agent: SACAgent,
    env: CBDCLiquidityEnv,
    n_episodes: int = 50,
) -> dict:
    """Evaluate an ablation agent over *n_episodes* deterministic episodes."""
    episode_rewards: list = []
    episode_funding_costs: list = []
    episode_lcr_violations: list = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        info: dict = {}

        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)
        episode_funding_costs.append(info.get("episode_funding_cost", 0.0))
        episode_lcr_violations.append(info.get("episode_lcr_violations", 0))

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_funding_cost": float(np.mean(episode_funding_costs)),
        "lcr_violation_rate": float(np.mean([v > 0 for v in episode_lcr_violations])),
    }


def generate_ablation_plots(df: pd.DataFrame, results_dir: Path) -> None:
    """Generate horizontal bar-chart and heatmap for ablation results."""
    sns.set_style("whitegrid")
    configs = list(df.index)
    n = len(configs)
    palette = sns.color_palette("RdYlGn", n)

    # ── Reward ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(4, n * 1.2)))
    ax.barh(configs, df["mean_reward"], color=palette, alpha=0.8)
    ax.set_xlabel("Mean Reward", fontsize=12)
    ax.set_title("Ablation: Mean Reward", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "ablation_reward.png", dpi=300)
    print(f"Saved: {results_dir / 'ablation_reward.png'}")
    plt.close()

    # ── Funding cost ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, max(4, n * 1.2)))
    ax.barh(configs, df["mean_funding_cost"], color=palette, alpha=0.8)
    ax.set_xlabel("Mean Funding Cost ($)", fontsize=12)
    ax.set_title("Ablation: Funding Cost", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "ablation_cost.png", dpi=300)
    print(f"Saved: {results_dir / 'ablation_cost.png'}")
    plt.close()

    # ── Multi-metric heatmap ──────────────────────────────────────────────────
    cols = [
        c
        for c in ["mean_reward", "mean_funding_cost", "lcr_violation_rate"]
        if c in df.columns
    ]
    if cols:
        df_norm = df[cols].copy()
        for col in cols:
            rng = df_norm[col].max() - df_norm[col].min()
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (rng + 1e-8)

        fig, ax = plt.subplots(figsize=(10, max(4, n * 1.2)))
        sns.heatmap(
            df_norm,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            ax=ax,
            cbar_kws={"label": "Normalized Score"},
        )
        ax.set_title("Ablation: Multi-Metric Heatmap", fontsize=14, fontweight="bold")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(results_dir / "ablation_heatmap.png", dpi=300)
        print(f"Saved: {results_dir / 'ablation_heatmap.png'}")
        plt.close()


if __name__ == "__main__":
    run_ablation_studies()
