# ==============================================================================
# Quantum-Enhanced Deep RL for CBDC Liquidity Optimization
# ==============================================================================
# Multi-stage build:
#   base    – Python 3.11 slim with system dependencies
#   deps    – pip dependencies installed (cached layer)
#   runtime – final image with project code
# ==============================================================================

# ── Stage 1: base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

LABEL maintainer="CBDC Research Team"
LABEL description="Quantum-Enhanced Deep RL for CBDC Liquidity Optimization"

# Prevent .pyc files and force stdout/stderr to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Keep pip quiet and fast
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Project directories
    PROJECT_ROOT=/app \
    PYTHONPATH=/app/code

WORKDIR /app

# System-level dependencies needed by PyTorch, PennyLane, Matplotlib, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: deps ─────────────────────────────────────────────────────────────
FROM base AS deps

# Copy only requirements first — Docker caches this layer until requirements change
COPY code/requirements.txt ./code/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r code/requirements.txt

# ── Stage 3: runtime ──────────────────────────────────────────────────────────
FROM deps AS runtime

# Copy the full project
COPY . .

# Create runtime directories (logs, MLflow artefacts, etc.)
RUN mkdir -p logs/metrics logs/plots logs/trained_models mlruns

# Install the package in editable mode so imports resolve without PYTHONPATH hacks
RUN pip install -e scripts/ --no-deps --quiet

# Non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --no-create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose MLflow tracking server port
EXPOSE 5000

# Health check — verify the environment can be instantiated
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from env.cbdc_env import CBDCLiquidityEnv; CBDCLiquidityEnv()" || exit 1

# Default: run verification script
CMD ["bash", "scripts/verify_installation.sh"]
