#!/bin/bash
# Quick verification and installation script

echo "========================================"
echo "Quantum CBDC Liquidity Project Verification"
echo "========================================"

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Install package
echo ""
echo "Installing package..."
pip install -e . -q

# Run tests
echo ""
echo "Running unit tests..."
pytest tests/ -v --tb=short

# Check imports
echo ""
echo "Verifying imports..."
python3 -c "
from env.cbdc_env import CBDCLiquidityEnv
from models.sac_agent import SACAgent
from models.vqc import VariationalQuantumCircuit
print('✓ All imports successful')
"

# Check environment
echo ""
echo "Testing environment..."
python3 -c "
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
python3 -c "
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
echo "To train models, run:"
echo "  python experiments/run_all_experiments.py"
echo ""
echo "To run individual training:"
echo "  python training/train_sac.py"
echo "  python training/train_qsac.py"
echo ""
echo "To evaluate models:"
echo "  python training/evaluate.py"
echo ""
echo "To analyze results:"
echo "  jupyter notebook notebooks/results_analysis.ipynb"
