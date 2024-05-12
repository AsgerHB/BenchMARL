#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    task: str = MISSING
    max_cycles: int = MISSING
    local_ratio: float = MISSING
    N: int = MISSING
