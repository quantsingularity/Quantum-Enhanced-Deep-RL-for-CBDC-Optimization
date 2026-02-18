# Quantum-Enhanced Deep Reinforcement Learning for CBDC Liquidity Management

A production-grade implementation of Quantum-Enhanced Soft Actor-Critic (QSAC) for optimizing Central Bank Digital Currency liquidity management under stress scenarios.

## Overview

This project implements a hybrid quantum-classical reinforcement learning framework that integrates Variational Quantum Circuits (VQC) into the Soft Actor-Critic algorithm to optimize commercial bank funding strategies under CBDC-induced liquidity stress.

## Key Features

- **Quantum-Enhanced SAC**: Hybrid quantum-classical critic network with 4-qubit VQC
- **Realistic CBDC Environment**: Stochastic liquidity dynamics with jump-diffusion CBDC shocks
- **Regulatory Constraints**: LCR enforcement and capital adequacy requirements
- **Comprehensive Benchmarking**: Classical SAC, QSAC, and rule-based baselines
- **Statistical Validation**: Paired t-tests, bootstrap CI, Wilcoxon tests
- **Full Reproducibility**: Fixed seeds, MLflow tracking, deterministic training

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, recommended)

### Setup

```bash
# Clone or extract the project
cd Quantum-Enhanced-Deep-RL-for-CBDC-Optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

## Quick Start

### 1. Train All Models

```bash
# Run complete experimental suite (Classical SAC, QSAC, Rule-based)
python experiments/run_all_experiments.py
```

This will:

- Train Classical SAC for 1M steps
- Train QSAC for 1M steps
- Evaluate rule-based baseline
- Generate comparison metrics
- Save results to `logs/`

### 2. Train Individual Models

```bash
# Train Classical SAC
python training/train_sac.py --config configs/sac.yaml

# Train Quantum-Enhanced SAC
python training/train_qsac.py --config configs/qsac.yaml
```

### 3. Evaluate Trained Models

```bash
# Evaluate all models
python training/evaluate.py --model_dir logs/trained_models/
```

### 4. Run Ablation Studies

```bash
python experiments/ablation_studies.py
```

### 5. Statistical Testing

```bash
python experiments/statistical_tests.py
```

### 6. Analyze Results

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/results_analysis.ipynb
```

## Project Structure

```
Quantum-Enhanced-Deep-RL-for-CBDC-Optimization/
│
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
│
├── configs/                    # Configuration files
│   ├── default.yaml           # Default parameters
│   ├── sac.yaml               # Classical SAC config
│   ├── qsac.yaml              # Quantum SAC config
│   └── environment.yaml       # Environment parameters
│
├── env/                        # CBDC Liquidity Environment
│   ├── cbdc_env.py            # Gymnasium environment
│   ├── liquidity_dynamics.py # Stochastic processes
│   ├── constraints.py         # LCR and capital constraints
│   └── reward.py              # Reward function
│
├── models/                     # Neural network models
│   ├── actor.py               # SAC actor network
│   ├── critic_classical.py    # Classical critic
│   ├── critic_quantum.py      # Quantum-enhanced critic
│   ├── vqc.py                 # Variational Quantum Circuit
│   └── sac_agent.py           # SAC algorithm implementation
│
├── training/                   # Training scripts
│   ├── train_sac.py           # Train classical SAC
│   ├── train_qsac.py          # Train quantum SAC
│   ├── evaluate.py            # Model evaluation
│   └── replay_buffer.py       # Experience replay
│
├── baselines/                  # Baseline policies
│   └── rule_based_policy.py   # Rule-based benchmark
│
├── experiments/                # Experimental suite
│   ├── run_all_experiments.py # Run all experiments
│   ├── ablation_studies.py    # Ablation analysis
│   └── statistical_tests.py   # Statistical validation
│
├── logs/                       # Experiment logs (generated)
│   ├── mlruns/                # MLflow tracking
│   ├── trained_models/        # Model checkpoints
│   ├── metrics/               # CSV metrics
│   └── plots/                 # Generated figures
│
├── notebooks/                  # Analysis notebooks
│   └── results_analysis.ipynb # Results visualization
│
└── tests/                      # Unit tests
    ├── test_environment.py
    ├── test_vqc.py
    └── test_sac.py
```

## Configuration

All hyperparameters are managed via YAML configs in `configs/`:

- `default.yaml`: Shared parameters (seeds, logging, etc.)
- `sac.yaml`: Classical SAC hyperparameters
- `qsac.yaml`: Quantum SAC hyperparameters (qubits, layers, etc.)
- `environment.yaml`: Environment dynamics and constraints

Edit these files to customize experiments.

## Reproducibility

All experiments use fixed random seeds for reproducibility:

```python
SEED = 42  # Set in configs/default.yaml
```

To reproduce paper results exactly:

```bash
# Run with default seed
python experiments/run_all_experiments.py

# Results will match published metrics within statistical variance
```

## Key Results

Expected performance improvements (QSAC vs Classical SAC):

- **Funding Cost Reduction**: ~8-12%
- **LCR Violation Reduction**: ~15-20%
- **Stability Index**: +10-15%
- **Statistical Significance**: p < 0.01 (paired t-test)

See `notebooks/results_analysis.ipynb` for detailed analysis.

## Environment Details

### State Space (8D)

1. Current liquidity buffer
2. Short-term liabilities
3. Projected inflows
4. Projected outflows
5. Interbank funding rate
6. CBDC demand shock
7. Market volatility proxy
8. Previous action

### Action Space (3D - Continuous)

1. Borrow amount [0, max_borrow]
2. Liquid asset reallocation ratio [0, 1]
3. Emergency funding decision [0, 1]

### Stochastic Dynamics

- **Inflows/Outflows**: Geometric Brownian Motion
- **Funding Rates**: Ornstein-Uhlenbeck process
- **CBDC Shocks**: Jump-diffusion with Poisson arrivals

### Constraints

- **LCR**: Liquidity Coverage Ratio ≥ 100%
- **Capital Adequacy**: Minimum capital buffer
- **Penalties**: Cost penalties for constraint violations

## Quantum Architecture

### VQC Design

- **Qubits**: 4 qubits (configurable)
- **Encoding**: RY rotation encoding
- **Entanglement**: CNOT ring topology
- **Parameterized Layers**: 2 layers of RY-RZ rotations
- **Measurement**: Pauli-Z expectation values
- **Backend**: PennyLane default.qubit (CPU) or qiskit (GPU)

### Noise Mitigation

- Zero Noise Extrapolation (ZNE) for hardware noise
- Configurable noise injection for testing robustness

## Advanced Usage

### Custom Training

```python
from training.train_qsac import train_qsac
from hydra import compose, initialize

# Initialize Hydra config
initialize(config_path="../configs", version_base=None)
cfg = compose(config_name="qsac")

# Train with custom parameters
train_qsac(cfg, n_steps=2000000, eval_freq=10000)
```

### Custom Environment

```python
from env.cbdc_env import CBDCLiquidityEnv

env = CBDCLiquidityEnv(
    initial_liquidity=1000000,
    lcr_threshold=1.0,
    cbdc_shock_intensity=0.3,
    seed=42
)
```

### Load Trained Model

```python
import torch
from models.sac_agent import SACAgent

# Load QSAC model
agent = SACAgent.load("logs/trained_models/qsac_final.pt")

# Evaluate
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.select_action(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
```

## Testing

Run unit tests:

```bash
# Test environment
pytest tests/test_environment.py -v

# Test VQC
pytest tests/test_vqc.py -v

# Test SAC agent
pytest tests/test_sac.py -v

# Run all tests
pytest tests/ -v
```

## Monitoring

### MLflow UI

```bash
# Start MLflow server
mlflow ui --backend-store-uri logs/mlruns

# Open browser at http://localhost:5000
```

### TensorBoard (optional)

```bash
tensorboard --logdir logs/tensorboard
```

## Performance Optimization

### GPU Acceleration

```python
# Automatically uses CUDA if available
# Force CPU:
export CUDA_VISIBLE_DEVICES=""

# Select specific GPU:
export CUDA_VISIBLE_DEVICES=0
```

### Parallel Evaluation

```python
# In configs/default.yaml
n_eval_envs: 8  # Vectorized environments for faster evaluation
```

## Troubleshooting

### Common Issues

1. **PennyLane installation fails**

   ```bash
   pip install pennylane --no-cache-dir
   ```

2. **CUDA out of memory**
   - Reduce batch_size in config
   - Reduce n_qubits in qsac.yaml

3. **Slow training**
   - Enable GPU acceleration
   - Reduce quantum circuit depth
   - Use classical SAC for faster iteration

4. **Numerical instability**
   - Adjust learning rates in config
   - Increase target_entropy for more exploration
   - Normalize observations

## License

MIT License - see LICENSE file for details
