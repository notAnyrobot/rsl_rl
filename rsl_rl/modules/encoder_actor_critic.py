# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.modules.actor_critic import ActorCritic

class EncoderActorCritic(ActorCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_encoder(self):
        """Build the encoder network."""
        raise NotImplementedError
