#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    n_agents: int = MISSING
    t_act: float = MISSING
    distance_min: float = MISSING
    distance_max: float = MISSING
    v_min: float = MISSING
    v_max: float = MISSING
    initial_distance: float = MISSING
    safety_violation_penalty: float = MISSING