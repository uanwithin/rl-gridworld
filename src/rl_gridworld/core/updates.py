"""算法更新通知；回调接收独立快照，不会修改算法内部状态。"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class Update:
    iteration: int
    phase: str
    before: np.ndarray
    after: np.ndarray


UpdateCallback = Callable[[Update], None]


def notify(
    callback: UpdateCallback | None,
    iteration: int,
    phase: str,
    before: np.ndarray,
    after: np.ndarray,
) -> None:
    if callback is not None:
        callback(Update(iteration, phase, before.copy(), after.copy()))
