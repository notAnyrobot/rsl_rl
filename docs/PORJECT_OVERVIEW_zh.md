# RSL-RL 项目概览

## 执行摘要

**RSL-RL** 是一个快速简洁的机器人强化学习算法实现，由苏黎世联邦理工学院（ETH Zurich）机器人系统实验室与 NVIDIA 合作开发。该库提供了生产就绪的最先进 RL 算法实现，包含适用于机器人应用的先进功能。

**当前版本：** 3.2.0  
**许可证：** BSD-3-Clause  
**维护者：** Mayank Mittal 和 Clemens Schwarke（ETH Zurich & NVIDIA）

---

## 项目架构

### 核心组件

该库组织为六个主要模块：

1. **算法** (`rsl_rl/algorithms/`)
   - **PPO（近端策略优化）**：主要的在线策略 RL 算法
   - **蒸馏（Distillation）**：学生-教师知识蒸馏框架

2. **模块** (`rsl_rl/modules/`)
   - **Actor-Critic 网络**：标准、基于 CNN 和循环变体
   - **学生-教师网络**：用于策略蒸馏
   - **随机网络蒸馏（RND）**：好奇心驱动的探索
   - **对称性增强**：基于对称性的数据增强和镜像损失

3. **网络** (`rsl_rl/networks/`)
   - **MLP**：多层感知器网络
   - **CNN**：用于基于视觉策略的卷积神经网络
   - **Memory**：用于循环策略的 RNN/LSTM 模块
   - **Normalization**：经验归一化层

4. **存储** (`rsl_rl/storage/`)
   - **RolloutStorage**：用于在线策略算法的高效缓冲区管理

5. **运行器** (`rsl_rl/runners/`)
   - **OnPolicyRunner**：训练和评估编排
   - **DistillationRunner**：学生-教师训练管道

6. **工具** (`rsl_rl/utils/`)
   - **Logger**：与 TensorBoard、Weights & Biases 和 Neptune 的集成
   - **Utilities**：用于配置和观察处理的辅助函数

---

## 主要功能

### 1. 近端策略优化（PPO）
- 带自适应学习率调度的裁剪代理目标
- 支持前馈、CNN 和循环策略架构
- 多 GPU 分布式训练支持
- 高级功能：
  - 裁剪价值损失
  - 每个小批量的优势归一化
  - 基于 KL 散度的自适应学习率

### 2. 随机网络蒸馏（RND）
- 好奇心驱动的内在奖励机制
- 可配置的权重调度（常数、步进、线性）
- 状态和奖励归一化支持
- 在稀疏奖励环境中鼓励探索

### 3. 基于对称性的增强
- 用于对称机器人行为的数据增强
- 用于强制执行对称性约束的镜像损失
- 可配置的增强函数
- 提高样本效率和泛化能力

### 4. 学生-教师蒸馏
- 从特权教师到学生策略的知识转移
- 支持前馈和循环架构
- 多种损失函数（MSE、Huber）
- 梯度累积以实现稳定训练

### 5. 多评论家支持（Dev 分支）
- 用于不同奖励组件的多个价值函数评论家
- 跨评论家组的加权优势计算
- 灵活的奖励组配置
- 增强复杂奖励结构的价值估计

### 6. L2C2 平滑性损失（Dev 分支）
- 策略和价值函数平滑性正则化
- 基于 Mixup 的插值用于平滑性计算
- 可配置的平滑性系数
- 提高策略稳定性和泛化能力

---

## `dev` 分支的开发

`dev` 分支包含 **95 个提交**（共 99 个提交），代表了活跃的开发线，包含多项重大增强：

### 近期主要功能（2025 年）

#### 1. 多评论家架构（提交：da16498）
**状态：** 积极开发中

**描述：** 实现多评论家价值函数架构，允许多个价值评论家估计不同的奖励组件。

**关键变更：**
- 修改 `ActorCritic` 以支持 `num_critics` 参数（默认：1）
- 更新 `RolloutStorage` 以处理多维奖励和价值
- 增强 PPO 算法以跨评论家组计算加权优势
- 添加 `value_group_weight` 参数用于自定义不同评论家的权重
- 修复奖励记录中的维度不匹配问题

**影响：**
- 为多目标任务启用更复杂的价值估计
- 支持复杂奖励结构的分解
- 在具有多个奖励信号的环境中提高学习稳定性

**修改的文件：**
- `rsl_rl/algorithms/ppo.py` (+28 行)
- `rsl_rl/modules/actor_critic.py` (+8 行)
- `rsl_rl/runners/on_policy_runner.py` (+19 行)
- `rsl_rl/storage/rollout_storage.py` (+13 行)

#### 2. L2C2 平滑性损失（提交：92d4b76）
**状态：** 初始实现

**描述：** 实现受 L2C2（Lipschitz 约束学习）方法启发的平滑性正则化损失。

**关键特性：**
- 策略平滑性损失：鼓励平滑的策略输出
- 价值平滑性损失：鼓励平滑的价值函数估计
- 基于 Mixup 的插值：在连续观察之间使用随机插值
- 可配置系数：策略和价值平滑性的单独权重
- 有界平滑性：平滑性计算的上限和下限

**实现细节：**
```python
# 平滑性损失计算
mix_weights = torch.rand(next_obs_batch.shape[0], device=device)
mix_obs_batch = obs_batch + mix_weights * (next_obs_batch - obs_batch)

policy_smooth_loss = ||μ(obs) - μ(mix_obs)||²
value_smooth_loss = ||V(obs) - V(mix_obs)||²
```

**影响：**
- 提高策略稳定性和泛化能力
- 减少对训练轨迹的过拟合
- 增强对观察扰动的鲁棒性

**修改的文件：**
- `rsl_rl/algorithms/ppo.py` (+31 行)
- `rsl_rl/storage/rollout_storage.py` (+21 行)

#### 3. 错误修复和改进

**近期修复：**
- **998dd76**：修复了在使用多评论家配置记录奖励时的维度不匹配问题
- **d41387e**：版本升级至 3.2.0
- **23a5b2a**：在训练日志中添加 `run_name` 以提高可追溯性
- **17bb7c0**：将日志功能从运行器中分离以提高模块化
- **6d6d3a4**：重构 rollout 存储以提高清晰度
- **e1e7071**：在轨迹分割中为 TensorDict 添加设备规范
- **c741f01**：移除不必要的教师评估调用
- **e8455fa**：添加用于基于视觉策略的感知 actor-critic 类

**代码质量改进：**
- **875b1da**：整个代码库的格式修复
- **8363520**：版本管理（v3.1.1、v3.1.2、v3.1.3）
- **530f71a**：修复 `act_inference` 在部署时返回策略均值而不包含标准差
- **1c63d8e**：修复教师观察归一化器输入大小不正确的问题

---

## 技术规格

### 依赖项
- **PyTorch**：>= 2.6.0
- **TensorDict**：>= 0.7.0
- **NumPy**：>= 1.16.4
- **ONNX/ONNXScript**：>= 0.5.4（用于模型导出）
- **Python**：>= 3.9

### 支持的环境
该库设计用于：
- **Isaac Lab**（NVIDIA Isaac Sim）
- **Legged Gym**（NVIDIA Isaac Gym）
- **MuJoCo Playground**（MuJoCo MJX 和 Warp）
- **mjlab**（MuJoCo Warp）

### 日志框架
- TensorBoard
- Weights & Biases (wandb)
- Neptune

### 代码质量工具
- **pre-commit**：自动化代码格式化和 linting
- **ruff**：快速的 Python linter 和格式化工具（用 Rust 编写）
- **pyright**：类型检查（基本模式）

---

## 研究贡献

该库实现了多项研究贡献：

1. **RSL-RL 库**（arXiv:2509.10771, 2025）
   - 描述框架的主要库论文

2. **好奇心驱动学习**（CoRL 2023）
   - 用于联合运动和操作的随机网络蒸馏

3. **对称性考虑**（ICRA 2024）
   - 用于学习任务对称机器人策略的基于对称性的增强

---

## 开发统计

- **总提交数**：99
- **Dev 分支提交数**：95
- **活跃贡献者**：多人（参见 CONTRIBUTORS.md）
- **版本历史**：
  - v3.2.0（当前，dev 分支）
  - v3.1.3
  - v3.1.2
  - v3.1.1

### 代码结构
- **算法**：2 个主要算法（PPO、Distillation）
- **网络架构**：3 种变体（标准、CNN、循环）
- **高级模块**：RND、Symmetry、Student-Teacher
- **代码行数**：约 5,000+（估计）

---

## Dev 分支的主要成就

1. **多评论家架构**：为复杂奖励结构启用复杂的价值估计
2. **L2C2 平滑性**：通过平滑性正则化提高策略稳定性和泛化能力
3. **增强的日志记录**：更好的可追溯性和模块化日志系统
4. **代码质量**：改进的代码组织、格式化和类型安全
5. **错误修复**：针对边缘情况和维度不匹配的多项关键修复

---

## 未来发展方向

基于当前 `dev` 分支的活动：

1. **多评论家优化**：进一步测试和优化多评论家架构
2. **L2C2 集成**：完成平滑性损失的集成和验证
3. **性能优化**：持续改进训练效率
4. **文档**：增强新功能的文档
5. **测试**：扩展新功能的测试覆盖率

---

## 使用示例

```python
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO

# 使用配置初始化运行器
runner = OnPolicyRunner(env, train_cfg, log_dir="./logs")

# 训练指定迭代次数
runner.learn(num_learning_iterations=1000)

# 评估训练的策略
runner.eval(num_eval_episodes=100)
```

---

## 引用

如果使用此库，请引用：

```bibtex
@article{schwarke2025rslrl,
  title={RSL-RL: A Learning Library for Robotics Research},
  author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
  journal={arXiv preprint arXiv:2509.10771},
  year={2025}
}
```

---

## 联系方式

**维护者：**
- Clemens Schwarke: cschwarke@ethz.ch
- Mayank Mittal: mittalma@ethz.ch

**代码库：** https://github.com/leggedrobotics/rsl_rl

---

*本文档为 KPI 报告目的生成。最后更新：2025 年 12 月*

