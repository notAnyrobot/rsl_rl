# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from rsl_rl.env import VecEnv


def resolve_l2c2_smoothness_config(alg_cfg: dict, env: VecEnv) -> dict:
    """Resolve the L2C2 smoothness configuration.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        env: Environment object.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    # If using L2C2 smoothness then pass the environment config object
    # Note: This is used by the L2C2 smoothness function for handling different observation terms
    if "l2c2_smoothness_cfg" in alg_cfg and alg_cfg["l2c2_smoothness_cfg"] is not None:
        alg_cfg["l2c2_smoothness_cfg"]["_env"] = env
    else:
        alg_cfg["l2c2_smoothness_cfg"] = None
    return alg_cfg
