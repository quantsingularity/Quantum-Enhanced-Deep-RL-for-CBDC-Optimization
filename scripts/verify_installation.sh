#!/bin/bash
# verify_installation.sh
# Run from the project root or from scripts/. Installs the package and
# confirms that the environment, models, and VQC are all functional.

set -euo pipefail

# ── resolve project root (works whether called from root or scripts/) ─────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "Quantum CBDC Liquidity Project Verification"
echo "========================================"
echo "Project root: $PROJECT_ROOT"

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Install package (setup.py lives in scripts/)
echo ""
echo "Installing package..."
pip install -e "$SCRIPT_DIR" -q

# Run tests
echo ""
echo "Running unit tests..."
pytest "$PROJECT_ROOT/code/tests/" -v --tb=short

# Check imports (PYTHONPATH set so bare module names resolve)
echo ""
echo "Verifying imports..."
PYTHONPATH="$PROJECT_ROOT/code" python3 -c "
from env.cbdc_env import CBDCLiquidityEnv
from models.sac_agent import SACAgent
from models.vqc import VariationalQuantumCircuit
print('✓ All imports successful')
"

# Check environment
echo ""
echo "Testing environment..."
PYTHONPATH="$PROJECT_ROOT/code" python3 -c "
from env.cbdc_env import CBDCLiquidityEnv
env = CBDCLiquidityEnv(seed=42)
obs, _ = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f'✓ Environment functional (reward: {reward:.2f})')
env.close()
"

# Check quantum circuit
echo ""
echo "Testing quantum circuit..."
PYTHONPATH="$PROJECT_ROOT/code" python3 -c "
import torch
from models.vqc import VariationalQuantumCircuit
vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)
inputs = torch.randn(1, 4) * 3.14
weights = vqc.init_weights()
output = vqc(inputs, weights)
print(f'✓ VQC functional (output shape: {output.shape})')
"

echo ""
echo "========================================"
echo "Verification Complete!"
echo "========================================"
echo ""
echo "To run all experiments:"
echo "  cd $PROJECT_ROOT"
echo "  PYTHONPATH=code python code/experiments/run_all_experiments.py"
echo ""
echo "To run individual training:"
echo "  PYTHONPATH=code python code/training/train_sac.py"
echo "  PYTHONPATH=code python code/training/train_qsac.py"
echo ""
echo "To evaluate a trained model:"
echo "  PYTHONPATH=code python code/training/evaluate.py"
