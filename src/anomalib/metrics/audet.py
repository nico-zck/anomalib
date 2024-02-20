"""Implementation of AUDET metric based on TorchMetrics."""
from typing import Optional, Tuple, Union

import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.functional.classification.precision_recall_curve import _binary_clf_curve
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.utilities.compute import auc, _auc_compute_without_check
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_curve

from .plotting_utils import plot_figure


# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class AUDET(BinaryPrecisionRecallCurve):
    """Area under the Detection error tradeoff (DET) curve.
    ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.det_curve.html

    Examples:
        >>> import torch
        >>> from anomalib.metrics import AUDET
        ...
        >>> preds = torch.tensor([0.13, 0.26, 0.08, 0.92, 0.03])
        >>> target = torch.tensor([0, 0, 1, 1, 0])
        ...
        >>> audet = AUDET()
        >>> audet(preds, target)
        tensor(0.6667)

        It is possible to update the metric state incrementally:

        >>> audet.update(preds[:2], target[:2])
        >>> audet.update(preds[2:], target[2:])
        >>> audet.compute()
        tensor(0.6667)

        To plot the Detection error tradeoff (DET) curve, use the ``generate_figure`` method:

        >>> fig, title = audet.generate_figure()
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = False
    full_state_update: bool = False

    def compute(self) -> torch.Tensor:
        """First compute Detection error tradeoff (DET) curve, then compute area under the curve.

        Returns:
            Tensor: Value of the AUDET metric
        """
        fnr: torch.Tensor
        fpr: torch.Tensor

        fpr, fnr, _ = self._compute()
        return auc(fpr, fnr, reorder=True)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with new values.

        Need to flatten new values as ROC expects them in this format for binary classification.

        Args:
            preds (torch.Tensor): predictions of the model
            target (torch.Tensor): ground truth targets
        """
        super().update(preds.flatten(), target.flatten())

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute fpr/fnr value pairs.

        Returns:
            Tuple containing Tensors for fpr and fnr
        """
        target = dim_zero_cat(self.target)
        preds = dim_zero_cat(self.preds)
        fpr, fnr, thres = det_curve(y_true=target, y_score=preds)
        return (fpr, fnr, thres)

    def generate_figure(self) -> tuple[Figure, str]:
        """Generate a figure containing the Detection error tradeoff (DET) curve, the baseline and the AUDET.

        Returns:
            tuple[Figure, str]: Tuple containing both the figure and the figure title to be used for logging
        """
        fpr, fnr, _ = self._compute()
        audet = self.compute()

        xlim = (-0.01, 1.01)
        ylim = (-0.01, 1.01)
        xlabel = "False Positive Rate (Overkill)"
        ylabel = "False Negative Rate (Misskill)"
        loc = "lower right"
        title = "Detection error tradeoff (DET) Curve"

        fig, axis = plot_figure(fpr, fnr, audet, xlim, ylim, xlabel, ylabel, loc, title)

        axis.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            figure=fig,
        )

        return fig, title

    def plot(self, curve: Optional[Tuple[Tensor, Tensor, Tensor]] = None, score: Optional[Union[Tensor, bool]] = None,
             ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        curve_computed = curve or self._compute()
        score = (
            _auc_compute_without_check(curve_computed[0], curve_computed[1], 1.0)
            if not curve and score is True
            else None
        )
        return plot_curve(
            curve_computed,
            score=score,
            ax=ax,
            label_names=("False positive rate (Overkill)", "False negative rate (Misskill)"),
            name=self.__class__.__name__,
        )


def det_curve(y_true: Tensor, y_score: Tensor, pos_label=1, sample_weight=None) \
        -> Tuple[Tensor, Tensor, Tensor]:
    """Compute error rates for different probability thresholds.

    .. note::
       This metric is used for evaluation of ranking and error tradeoffs of
       a binary classification task.

    Read more in the :ref:`User Guide <det_curve>`.

    .. versionadded:: 0.24

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : ndarray of shape of (n_samples,)
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int, float, bool or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if `y_true` is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fpr : ndarray of shape (n_thresholds,)
        False positive rate (FPR) such that element i is the false positive
        rate of predictions with score >= thresholds[i]. This is occasionally
        referred to as false acceptance probability or fall-out.

    fnr : ndarray of shape (n_thresholds,)
        False negative rate (FNR) such that element i is the false negative
        rate of predictions with score >= thresholds[i]. This is occasionally
        referred to as false rejection or miss rate.

    thresholds : ndarray of shape (n_thresholds,)
        Decreasing score values.

    See Also
    --------
    DetCurveDisplay.from_estimator : Plot DET curve given an estimator and
        some data.
    DetCurveDisplay.from_predictions : Plot DET curve given the true and
        predicted labels.
    DetCurveDisplay : DET curve visualization.
    roc_curve : Compute Receiver operating characteristic (ROC) curve.
    precision_recall_curve : Compute precision-recall curve.
    """
    fps, tps, thresholds = _binary_clf_curve(
        preds=y_score, target=y_true, pos_label=pos_label, sample_weights=sample_weight
    )

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Only one class present in y_true. Detection error "
            "tradeoff curve is not defined in that case."
        )

    fns = tps[-1] - tps
    p_count = tps[-1]
    n_count = fps[-1]

    # disabling dropping intermediate values
    # # start with false positives zero
    # first_ind = (
    #     fps.searchsorted(fps[0], side="right") - 1
    #     if fps.searchsorted(fps[0], side="right") > 0
    #     else None
    # )
    # # stop with false negatives zero
    # last_ind = tps.searchsorted(tps[-1]) + 1
    # sl = slice(first_ind, last_ind)
    #
    # # reverse the output such that list of false positives is decreasing
    # return (fps[sl][::-1] / n_count, fns[sl][::-1] / p_count, thresholds[sl][::-1])

    return (fps.flip(-1) / n_count, fns.flip(-1) / p_count, thresholds.flip(-1))
