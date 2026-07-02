# Quantum-Enhanced Deep RL for CBDC Liquidity Optimization

A research project combining **Variational Quantum Circuits (VQC)** with **Soft Actor-Critic (SAC)** reinforcement learning to optimize liquidity management in Central Bank Digital Currency (CBDC) systems.

---

## Repository structure

```
.
├── .github/
│   └── workflows/
│       └── cicd.yml                 # CI: formatting checks (autoflake, black, prettier)
├── code/                            # All Python source code
│   ├── baselines/                   # Rule-based policy baseline
│   ├── env/                         # CBDC Gymnasium environment
│   ├── experiments/                 # Ablation studies, statistical tests, full suite
│   ├── models/                      # Actor, critics (classical + quantum), SAC, VQC
│   ├── tests/                       # Unit tests (environment, SAC, VQC)
│   ├── training/                    # Training loops and evaluation utilities
│   ├── conftest.py                  # pytest sys.path bootstrap
│   └── requirements.txt             # Python dependencies
├── docs/
│   └── ci-formatting-check.md       # CI workflow documentation
├── infrastructure/
│   └── configs/                     # YAML configs (default, environment, sac, qsac)
├── scripts/
│   ├── lint.sh                      # Interactive linting (ruff, flake8, mypy, pylint)
│   ├── run.sh                       # Single entry point: setup/test/train/evaluate/demo
│   ├── setup.py                     # Package installation
│   └── verify_installation.sh       # End-to-end installation & smoke-test
├── Dockerfile                       # Multi-stage Docker build
├── docker-compose.yml               # Orchestrates app + MLflow + test services
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quick start: run script

The simplest way to use the repository is through the unified run script,
which handles PYTHONPATH and working directories for you:

```bash
bash scripts/run.sh setup        # install dependencies
bash scripts/run.sh test         # run the unit test suite (18 tests)
bash scripts/run.sh demo         # 1-minute smoke run: SAC training + baseline
bash scripts/run.sh train-sac    # train the classical SAC agent
bash scripts/run.sh train-qsac   # train the quantum-enhanced SAC agent
bash scripts/run.sh baseline     # evaluate the rule-based baseline
bash scripts/run.sh evaluate     # compare trained models
bash scripts/run.sh experiments  # full experimental suite
```

## Quick start: local

### 1. Install dependencies

```bash
pip install -r code/requirements.txt
```

### 2. Verify installation

```bash
bash scripts/verify_installation.sh
```

### 3. Train the classical SAC agent

```bash
PYTHONPATH=code python code/training/train_sac.py
```

### 4. Train the Quantum-SAC agent

```bash
PYTHONPATH=code python code/training/train_qsac.py
```

### 5. Run the full experimental suite

```bash
PYTHONPATH=code python code/experiments/run_all_experiments.py
```

### 6. Evaluate a trained model

```bash
PYTHONPATH=code python code/training/evaluate.py
```

---

## Quick start: Docker

### Build the image

```bash
docker compose build
```

### Start MLflow tracking server

```bash
docker compose up mlflow -d
# UI at http://localhost:5000
```

### Run the full experiment suite

```bash
docker compose up app
```

### Run tests only

```bash
docker compose --profile test up tests
```

### Train classical SAC only

```bash
docker compose --profile train-sac up train-sac
```

### Train quantum SAC only

```bash
docker compose --profile train-qsac up train-qsac
```

---

## Configuration

All YAML configs live in `infrastructure/configs/`:

| File               | Purpose                                      |
| ------------------ | -------------------------------------------- |
| `default.yaml`     | Seed, device, logging intervals, MLflow URI  |
| `environment.yaml` | CBDC environment dynamics parameters         |
| `sac.yaml`         | Classical SAC hyperparameters                |
| `qsac.yaml`        | Quantum-SAC hyperparameters + circuit config |

---

## Running tests

```bash
# With PYTHONPATH (conftest.py handles this automatically)
pytest code/tests/ -v --tb=short
```

---

## Linting

```bash
bash scripts/lint.sh
```

Runs ruff, flake8, mypy, and pylint against any sub-directory of `code/`.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- PennyLane ≥ 0.33
- Gymnasium ≥ 0.29
- MLflow ≥ 2.8

See `code/requirements.txt` for the full list.

---

## License

MIT: see [LICENSE](LICENSE).
