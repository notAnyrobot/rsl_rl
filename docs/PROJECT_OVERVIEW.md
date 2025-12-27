# RSL-RL Project Overview

## Executive Summary

**RSL-RL** is a fast and simple implementation of reinforcement learning algorithms for robotics, developed by the Robotic Systems Lab at ETH Zurich in collaboration with NVIDIA. The library provides production-ready implementations of state-of-the-art RL algorithms with advanced features for robotics applications.

**Current Version:** 3.2.0  
**License:** BSD-3-Clause  
**Maintainers:** Mayank Mittal and Clemens Schwarke (ETH Zurich & NVIDIA)

---

## Project Architecture

### Core Components

The library is organized into six main modules:

1. **Algorithms** (`rsl_rl/algorithms/`)
   - **PPO (Proximal Policy Optimization)**: Primary on-policy RL algorithm
   - **Distillation**: Student-teacher knowledge distillation framework

2. **Modules** (`rsl_rl/modules/`)
   - **Actor-Critic Networks**: Standard, CNN-based, and Recurrent variants
   - **Student-Teacher Networks**: For policy distillation
   - **Random Network Distillation (RND)**: Curiosity-driven exploration
   - **Symmetry Augmentation**: Symmetry-based data augmentation and mirror loss

3. **Networks** (`rsl_rl/networks/`)
   - **MLP**: Multi-layer perceptron networks
   - **CNN**: Convolutional neural networks for vision-based policies
   - **Memory**: RNN/LSTM modules for recurrent policies
   - **Normalization**: Empirical normalization layers

4. **Storage** (`rsl_rl/storage/`)
   - **RolloutStorage**: Efficient buffer management for on-policy algorithms

5. **Runners** (`rsl_rl/runners/`)
   - **OnPolicyRunner**: Training and evaluation orchestration
   - **DistillationRunner**: Student-teacher training pipeline

6. **Utils** (`rsl_rl/utils/`)
   - **Logger**: Integration with TensorBoard, Weights & Biases, and Neptune
   - **Utilities**: Helper functions for configuration and observation processing

---

## Key Features

### 1. Proximal Policy Optimization (PPO)
- Clipped surrogate objective with adaptive learning rate scheduling
- Support for feedforward, CNN, and recurrent policy architectures
- Multi-GPU distributed training support
- Advanced features:
  - Clipped value loss
  - Per-minibatch advantage normalization
  - KL-divergence-based adaptive learning rate

### 2. Random Network Distillation (RND)
- Curiosity-driven intrinsic reward mechanism
- Configurable weight scheduling (constant, step, linear)
- State and reward normalization support
- Encourages exploration in sparse reward environments

### 3. Symmetry-Based Augmentation
- Data augmentation for symmetric robot behaviors
- Mirror loss for enforcing symmetry constraints
- Configurable augmentation functions
- Improves sample efficiency and generalization

### 4. Student-Teacher Distillation
- Knowledge transfer from privileged teacher to student policy
- Support for both feedforward and recurrent architectures
- Multiple loss functions (MSE, Huber)
- Gradient accumulation for stable training

### 5. Multi-Critic Support (Dev Branch)
- Multiple value function critics for different reward components
- Weighted advantage computation across critic groups
- Flexible reward group configuration
- Enhanced value estimation for complex reward structures

### 6. L2C2 Smoothness Loss (Dev Branch)
- Policy and value function smoothness regularization
- Mixup-based interpolation for smoothness computation
- Configurable smoothness coefficients
- Improves policy stability and generalization

---

## Development in `dev` Branch

The `dev` branch contains **95 commits** (out of 99 total commits) and represents the active development line with several significant enhancements:

### Recent Major Features (2025)

#### 1. Multi-Critic Architecture (Commit: da16498)
**Status:** Active Development

**Description:** Implementation of multi-critic value function architecture allowing multiple value critics to estimate different reward components.

**Key Changes:**
- Modified `ActorCritic` to support `num_critics` parameter (default: 1)
- Updated `RolloutStorage` to handle multi-dimensional rewards and values
- Enhanced PPO algorithm to compute weighted advantages across critic groups
- Added `value_group_weight` parameter for custom weighting of different critics
- Fixed dimension mismatch issues in reward logging

**Impact:**
- Enables more sophisticated value estimation for multi-objective tasks
- Supports decomposition of complex reward structures
- Improves learning stability in environments with multiple reward signals

**Files Modified:**
- `rsl_rl/algorithms/ppo.py` (+28 lines)
- `rsl_rl/modules/actor_critic.py` (+8 lines)
- `rsl_rl/runners/on_policy_runner.py` (+19 lines)
- `rsl_rl/storage/rollout_storage.py` (+13 lines)

#### 2. L2C2 Smoothness Loss (Commit: 92d4b76)
**Status:** Initial Implementation

**Description:** Implementation of smoothness regularization loss inspired by L2C2 (Lipschitz-constrained learning) methodology.

**Key Features:**
- Policy smoothness loss: Encourages smooth policy outputs
- Value smoothness loss: Encourages smooth value function estimates
- Mixup-based interpolation: Uses random interpolation between consecutive observations
- Configurable coefficients: Separate weights for policy and value smoothness
- Bounded smoothness: Upper and lower bounds for smoothness computation

**Implementation Details:**
```python
# Smoothness loss computation
mix_weights = torch.rand(next_obs_batch.shape[0], device=device)
mix_obs_batch = obs_batch + mix_weights * (next_obs_batch - obs_batch)

policy_smooth_loss = ||μ(obs) - μ(mix_obs)||²
value_smooth_loss = ||V(obs) - V(mix_obs)||²
```

**Impact:**
- Improves policy stability and generalization
- Reduces overfitting to training trajectories
- Enhances robustness to observation perturbations

**Files Modified:**
- `rsl_rl/algorithms/ppo.py` (+31 lines)
- `rsl_rl/storage/rollout_storage.py` (+21 lines)

#### 3. Bug Fixes and Improvements

**Recent Fixes:**
- **998dd76**: Fixed dimension mismatch issue when logging rewards with multi-critic configuration
- **d41387e**: Version bump to 3.2.0
- **23a5b2a**: Added `run_name` to training logs for better traceability
- **17bb7c0**: Separated logging functionality from runner for better modularity
- **6d6d3a4**: Restructured rollout storage for improved clarity
- **e1e7071**: Added device specification for TensorDict in trajectory splitting
- **c741f01**: Removed unnecessary teacher evaluation call
- **e8455fa**: Added perceptive actor-critic class for vision-based policies

**Code Quality Improvements:**
- **875b1da**: Formatting fixes across codebase
- **8363520**: Version management (v3.1.1, v3.1.2, v3.1.3)
- **530f71a**: Fixed `act_inference` to return policy mean without std dev at deployment
- **1c63d8e**: Fixed incorrect teacher observation normalizer input size

---

## Technical Specifications

### Dependencies
- **PyTorch**: >= 2.6.0
- **TensorDict**: >= 0.7.0
- **NumPy**: >= 1.16.4
- **ONNX/ONNXScript**: >= 0.5.4 (for model export)
- **Python**: >= 3.9

### Supported Environments
The library is designed to work with:
- **Isaac Lab** (NVIDIA Isaac Sim)
- **Legged Gym** (NVIDIA Isaac Gym)
- **MuJoCo Playground** (MuJoCo MJX and Warp)
- **mjlab** (MuJoCo Warp)

### Logging Frameworks
- TensorBoard
- Weights & Biases (wandb)
- Neptune

### Code Quality Tools
- **pre-commit**: Automated code formatting and linting
- **ruff**: Fast Python linter and formatter (written in Rust)
- **pyright**: Type checking (basic mode)

---

## Research Contributions

The library implements several research contributions:

1. **RSL-RL Library** (arXiv:2509.10771, 2025)
   - Main library paper describing the framework

2. **Curiosity-Driven Learning** (CoRL 2023)
   - Random Network Distillation for joint locomotion and manipulation

3. **Symmetry Considerations** (ICRA 2024)
   - Symmetry-based augmentation for learning task-symmetric robot policies

---

## Development Statistics

- **Total Commits**: 99
- **Dev Branch Commits**: 95
- **Active Contributors**: Multiple (see CONTRIBUTORS.md)
- **Version History**: 
  - v3.2.0 (current, dev branch)
  - v3.1.3
  - v3.1.2
  - v3.1.1

### Code Structure
- **Algorithms**: 2 main algorithms (PPO, Distillation)
- **Network Architectures**: 3 variants (Standard, CNN, Recurrent)
- **Advanced Modules**: RND, Symmetry, Student-Teacher
- **Lines of Code**: ~5,000+ (estimated)

---

## Key Achievements in Dev Branch

1. **Multi-Critic Architecture**: Enables sophisticated value estimation for complex reward structures
2. **L2C2 Smoothness**: Improves policy stability and generalization through smoothness regularization
3. **Enhanced Logging**: Better traceability and modular logging system
4. **Code Quality**: Improved code organization, formatting, and type safety
5. **Bug Fixes**: Multiple critical fixes for edge cases and dimension mismatches

---

## Future Development Directions

Based on the current `dev` branch activity:

1. **Multi-Critic Refinement**: Further testing and optimization of multi-critic architecture
2. **L2C2 Integration**: Complete integration and validation of smoothness loss
3. **Performance Optimization**: Continued improvements to training efficiency
4. **Documentation**: Enhanced documentation for new features
5. **Testing**: Expanded test coverage for new functionality

---

## Usage Example

```python
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO

# Initialize runner with configuration
runner = OnPolicyRunner(env, train_cfg, log_dir="./logs")

# Train for specified iterations
runner.learn(num_learning_iterations=1000)

# Evaluate trained policy
runner.eval(num_eval_episodes=100)
```

---

## Citation

If using this library, please cite:

```bibtex
@article{schwarke2025rslrl,
  title={RSL-RL: A Learning Library for Robotics Research},
  author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
  journal={arXiv preprint arXiv:2509.10771},
  year={2025}
}
```

---

## Contact

**Maintainers:**
- Clemens Schwarke: cschwarke@ethz.ch
- Mayank Mittal: mittalma@ethz.ch

**Repository:** https://github.com/leggedrobotics/rsl_rl

---

*Document generated for KPI reporting purposes. Last updated: December 2025*

