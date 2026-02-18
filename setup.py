from setuptools import setup, find_packages

setup(
    name="quantum_cbdc_liquidity",
    version="1.0.0",
    author="Abrar Ahmed",
    author_email="research@example.com",
    description="Quantum-Enhanced Deep RL for CBDC Liquidity Management",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum_cbdc_liquidity",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "pennylane>=0.33.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "gymnasium>=0.29.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "hydra-core>=1.3.0",
        "mlflow>=2.8.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "quantum": [
            "pennylane-qiskit>=0.33.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipython>=8.12.0",
        ],
    },
)
