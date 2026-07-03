#!/usr/bin/env bash
# =============================================================================
# Run script for Quantum-Enhanced Deep RL for CBDC Liquidity Optimization
#
# Usage:
#   bash scripts/run.sh <command> [options]
#
# Commands:
#   test         Run the unit test suite
#   train-sac    Train the classical SAC agent
#   train-qsac   Train the quantum-enhanced SAC agent
#   baseline     Evaluate the rule-based baseline policy
#   evaluate     Evaluate trained models and produce comparison CSV
#   experiments  Run the full experiment suite (ablations + statistics)
#   demo         Quick smoke run: short training + baseline evaluation
#   all          test -> train-sac -> train-qsac -> evaluate
#
# Extra arguments after the command are forwarded to the underlying
# Python entry point, e.g.:
#   bash scripts/run.sh train-sac            # full training
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$ROOT_DIR/code"

export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"

PYTHON="${PYTHON:-python3}"

usage() {
    awk 'NR>1 && /^# ={10,}/{c++; if(c==2) exit} NR>1{sub(/^# ?/,""); print}' "$0"
}

cmd="${1:-}"
shift || true

case "$cmd" in
    test)
        echo "[run.sh] Running unit tests..."
        cd "$CODE_DIR"
        "$PYTHON" -m pytest tests/ -v "$@"
        ;;

    train-sac)
        echo "[run.sh] Training classical SAC agent..."
        cd "$ROOT_DIR"
        "$PYTHON" "$CODE_DIR/training/train_sac.py" "$@"
        ;;

    train-qsac)
        echo "[run.sh] Training quantum-enhanced SAC agent..."
        cd "$ROOT_DIR"
        "$PYTHON" "$CODE_DIR/training/train_qsac.py" "$@"
        ;;

    baseline)
        echo "[run.sh] Evaluating rule-based baseline..."
        cd "$ROOT_DIR"
        "$PYTHON" "$CODE_DIR/baselines/rule_based_policy.py" "$@"
        ;;

    evaluate)
        echo "[run.sh] Evaluating trained models..."
        cd "$ROOT_DIR"
        "$PYTHON" "$CODE_DIR/training/evaluate.py" "$@"
        ;;

    experiments)
        echo "[run.sh] Running full experiment suite..."
        cd "$ROOT_DIR"
        "$PYTHON" "$CODE_DIR/experiments/run_all_experiments.py" "$@"
        ;;

    demo)
        echo "[run.sh] Demo: environment + agents smoke run (~1 min)..."
        cd "$CODE_DIR"
        "$PYTHON" - <<'PY'
import yaml
from pathlib import Path
from env.cbdc_env import CBDCLiquidityEnv
from models.sac_agent import SACAgent
from training.replay_buffer import ReplayBuffer
from baselines.rule_based_policy import evaluate_rule_based

cfg_dir = Path(__file__ if '__file__' in dir() else '.').resolve()
configs = Path.cwd().parent / "infrastructure" / "configs"
env_config = yaml.safe_load(open(configs / "environment.yaml"))

env = CBDCLiquidityEnv(seed=42, max_episode_steps=50, **env_config)
sd, ad = env.observation_space.shape[0], env.action_space.shape[0]
agent = SACAgent(state_dim=sd, action_dim=ad, device="cpu")
buf = ReplayBuffer(sd, ad, max_size=5000)

state, _ = env.reset()
for step in range(300):
    action = env.action_space.sample() if step < 50 else agent.select_action(state)
    nxt, reward, term, trunc, info = env.step(action)
    buf.add(state, action, reward, nxt, float(term))
    state = nxt
    if term or trunc:
        state, _ = env.reset()
    if step >= 50:
        agent.update(*buf.sample(32))

print("SAC demo training completed. Last LCR: %.3f" % info["lcr"])

env2 = CBDCLiquidityEnv(seed=7, max_episode_steps=50, **env_config)
metrics = evaluate_rule_based(env2, n_episodes=10)
print("Rule-based baseline over 10 episodes:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
print("\nDemo finished successfully.")
PY
        ;;

    all)
        bash "$0" test
        bash "$0" train-sac
        bash "$0" train-qsac
        bash "$0" evaluate
        ;;

    -h|--help|help|"")
        usage
        ;;

    *)
        echo "Unknown command: $cmd" >&2
        usage
        exit 1
        ;;
esac
