# GTP-SCUC: Graph Neural Networks & Reinforcement Learning for Fast SCUC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg)](https://pytorch.org/)

This repository contains the official open-source PyTorch implementation of the **GTP (GAN-TGNN-PPO)** framework, corresponding to our submitted manuscript: *"Fast solution method for mid- to long-term SCUC scheduling using graph neural networks and reinforcement learning"*.

To ensure methodological transparency and facilitate external reproducibility, this repository provides the core neural network architectures and the reinforcement learning environment wrappers. It includes the configuration for the standardized **IEEE 118-bus test system** benchmark discussed in **Section 5.11** of the manuscript.

## 🧠 Methodology Architecture

The GTP framework addresses the curse of dimensionality in Security-Constrained Unit Commitment (SCUC) through an asymmetric "screening-projection" dual-track architecture:
1. **Physics-Constrained CGAN (`models/gan.py`)**: Synthesizes Out-of-Distribution (OoD) extreme scenarios while maintaining linearized power balance constraints.
2. **Temporal Graph Neural Network (`models/tgnn.py`)**: Captures complex spatiotemporal topology via Graph Attention Networks (GAT) and dynamic temporal evolution via GRU, featuring a high-sensitivity dynamic fault mask.
3. **PPO with DAST (`models/ppo_agent.py`)**: Decouples the raw neural policy generation from the optimization-based LP repair layer. Utilizes Dynamic Affine Scaling Transformation (DAST) for continuous bounds.

## 🚀 Repository Structure
* `models/` - Core neural architectures (CGAN, TGNN, PPO Actor/Critic).
* `env/` - SCUC Environment interface demonstrating the evaluation-transition decoupling and VOLL soft pull-back mechanism.
* `main_ieee118.py` - Standardized execution script for the IEEE 118-bus benchmark.

## ⚙️ Environment Setup
The codebase requires `Python 3.9+`, `PyTorch 2.1.0`, and `PyTorch Geometric 2.4.0`. The LP projection layer interfaces natively with standard MILP solvers (e.g., Gurobi 10.0 or open-source equivalents like SciPy/HiGHS).

```bash
pip install torch torchvision torchaudio
pip install torch_geometric

## ⚠️ Disclaimer

This repository provides the core architectural implementation to ensure methodological transparency. Full reproduction of the exact baseline results requires access to commercial MILP solvers (e.g., Gurobi) for the LP projection layer and the complete year-long proprietary dataset. For open-source evaluation, we provide the dummy-based evaluation script `main_ieee118.py` to verify latency claims.
