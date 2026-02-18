"""
Train classical SAC agent.
"""

import yaml
import torch
import numpy as np
from tqdm import tqdm
import mlflow
from pathlib import Path

from env.cbdc_env import CBDCLiquidityEnv
from models.sac_agent import SACAgent
from training.replay_buffer import ReplayBuffer


def train_sac(
    config_path: str = "configs/sac.yaml",
    env_config_path: str = "configs/environment.yaml",
    default_config_path: str = "configs/default.yaml",
    n_steps: int = 1000000,
    eval_freq: int = 10000,
    n_eval_episodes: int = 100,
):
    """
    Train classical SAC agent.

    Args:
        config_path: Path to SAC config
        env_config_path: Path to environment config
        default_config_path: Path to default config
        n_steps: Total training steps
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
    """
    # Load configs
    with open(config_path, "r") as f:
        sac_config = yaml.safe_load(f)

    with open(env_config_path, "r") as f:
        env_config = yaml.safe_load(f)

    with open(default_config_path, "r") as f:
        default_config = yaml.safe_load(f)

    # Set seeds
    seed = default_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = default_config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create directories
    log_dir = Path(default_config["log_dir"])
    checkpoint_dir = log_dir / "trained_models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    mlflow.set_tracking_uri(default_config["mlflow_tracking_uri"])
    mlflow.set_experiment(default_config["experiment_name"])

    # Start MLflow run
    with mlflow.start_run(run_name="classical_sac"):
        # Log parameters
        mlflow.log_params(sac_config)
        mlflow.log_params({"model_type": "classical_sac", "seed": seed})

        # Create environment
        env = CBDCLiquidityEnv(seed=seed, **env_config)
        eval_env = CBDCLiquidityEnv(seed=seed + 1, **env_config)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Create agent
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            actor_hidden_dims=tuple(sac_config["actor_hidden_dims"]),
            critic_hidden_dims=tuple(sac_config["critic_hidden_dims"]),
            lr_actor=sac_config["learning_rate_actor"],
            lr_critic=sac_config["learning_rate_critic"],
            lr_alpha=sac_config["learning_rate_alpha"],
            gamma=sac_config["gamma"],
            tau=sac_config["tau"],
            auto_entropy_tuning=sac_config["auto_entropy_tuning"],
            device=device,
            use_quantum_critic=False,
        )

        # Create replay buffer
        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=sac_config["buffer_size"],
            device=device,
        )

        # Training loop
        state, _ = env.reset()
        episode_reward = 0
        episode_step = 0
        episode_num = 0
        best_eval_reward = -np.inf

        print("Starting training...")
        pbar = tqdm(total=n_steps, desc="Training SAC")

        for step in range(n_steps):
            # Select action
            if step < sac_config["learning_starts"]:
                # Random action for warmup
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
            episode_step += 1

            # Update agent
            if step >= sac_config["learning_starts"]:
                for _ in range(sac_config["gradient_steps"]):
                    batch = replay_buffer.sample(sac_config["batch_size"])
                    losses = agent.update(*batch)

                    # Log losses
                    if step % default_config["log_interval"] == 0:
                        for key, value in losses.items():
                            mlflow.log_metric(f"train/{key}", value, step=step)

            # Episode end
            if done:
                mlflow.log_metric("train/episode_reward", episode_reward, step=step)
                mlflow.log_metric("train/episode_length", episode_step, step=step)
                mlflow.log_metric("train/episode_num", episode_num, step=step)

                state, _ = env.reset()
                episode_reward = 0
                episode_step = 0
                episode_num += 1

            # Evaluation
            if (step + 1) % eval_freq == 0:
                eval_metrics = evaluate(agent, eval_env, n_eval_episodes)

                print(f"\nStep {step + 1} Evaluation:")
                print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f}")
                print(f"  Mean Funding Cost: ${eval_metrics['mean_funding_cost']:.2f}")
                print(f"  LCR Violation Rate: {eval_metrics['lcr_violation_rate']:.2%}")

                # Log evaluation metrics
                for key, value in eval_metrics.items():
                    mlflow.log_metric(f"eval/{key}", value, step=step)

                # Save best model
                if eval_metrics["mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_metrics["mean_reward"]
                    agent.save(str(checkpoint_dir / "sac_best.pt"))
                    print(f"  New best model saved!")

            # Save checkpoint
            if (step + 1) % default_config["save_freq"] == 0:
                agent.save(str(checkpoint_dir / f"sac_step_{step + 1}.pt"))

            pbar.update(1)

        pbar.close()

        # Save final model
        agent.save(str(checkpoint_dir / "sac_final.pt"))
        print(
            f"\nTraining complete! Final model saved to {checkpoint_dir / 'sac_final.pt'}"
        )

        # Final evaluation
        final_eval_metrics = evaluate(agent, eval_env, n_eval_episodes)
        print("\nFinal Evaluation:")
        for key, value in final_eval_metrics.items():
            print(f"  {key}: {value}")
            mlflow.log_metric(f"final/{key}", value)

        env.close()
        eval_env.close()


def evaluate(agent: SACAgent, env: CBDCLiquidityEnv, n_episodes: int = 100):
    """
    Evaluate agent.

    Args:
        agent: SAC agent
        env: Environment
        n_episodes: Number of episodes

    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_funding_costs = []
    episode_lcr_violations = []
    episode_lengths = []

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
        episode_lengths.append(info["step"])

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_funding_cost": np.mean(episode_funding_costs),
        "std_funding_cost": np.std(episode_funding_costs),
        "lcr_violation_rate": np.mean([v > 0 for v in episode_lcr_violations]),
        "mean_lcr_violations": np.mean(episode_lcr_violations),
        "mean_episode_length": np.mean(episode_lengths),
    }


if __name__ == "__main__":
    train_sac()
