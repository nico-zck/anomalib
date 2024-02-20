"""Implementation of OverKill and MissKill metrics based on TorchMetrics."""
from typing import Optional, Union, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics.classification import BinaryStatScores
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class OverKill(BinaryStatScores):
    """
    False positive rate (FPR), also known as OverKill Rate
    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "OverKill"

    def compute(self) -> Tensor:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        return _safe_divide(fp, fp + tn)

    def plot(
            self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        return self._plot(val, ax)

    def generate_figure(self) -> tuple[Figure, str]:
        """Generate a figure of OverKill(FPR).

        Returns:
            tuple[Figure, str]: Tuple containing both the figure and the figure title to be used for logging
        """

        title = "OverKill(FPR)"
        fig, axis = plt.subplots()
        self._plot(val=None, ax=axis)
        return fig, title


class MissKill(BinaryStatScores):
    """
    False negative rate (FNR), also known as MissKill Rate
    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "MissKill"

    def compute(self) -> Tensor:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        return _safe_divide(fn, tp + fn)

    def plot(
            self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        return self._plot(val, ax)

    def generate_figure(self) -> tuple[Figure, str]:
        """Generate a figure of MissKill(FNR).

        Returns:
            tuple[Figure, str]: Tuple containing both the figure and the figure title to be used for logging
        """

        title = "MissKill(FNR)"
        fig, axis = plt.subplots()
        self._plot(val=None, ax=axis)
        return fig, title
