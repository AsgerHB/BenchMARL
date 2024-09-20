#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    t_act: float = MISSING
    min_stored: float = MISSING
    max_stored: float = MISSING
    flow_rate: float = MISSING
    flow_variance: float = MISSING
    stored_initially: float = MISSING
    safety_violation_penalty: float = MISSING
    