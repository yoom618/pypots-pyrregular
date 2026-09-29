"""MASCIT train-loss-only wrapper.

This variant follows the "train-loss-only" checkpoint policy used in the reference
experimental setup: the same training set is used for validation, and the model
selects the best checkpoint according to the training loss curve rather than a
separate validation metric.
"""

from __future__ import annotations

from typing import Optional, Union

try:
    from pyrregular.conversion_utils import to_pypots
except ImportError:
    # PYRREGULAR renamed this private adapter in newer revisions.
    from pyrregular.conversion_utils import _to_pypots as to_pypots
from pyrregular.wrappers.pypots_wrapper import PyPOTSWrapper

from .model import MASCIT


class MASCITTrainLossOnlyWrapper(PyPOTSWrapper):
    """Train-loss-only wrapper for MASCIT.

    Parameters
    ----------
    model : type
        The MASCIT model class.
    model_params : dict
        Model config, including train-only checkpoint settings.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, model, model_params, random_state=None):
        super().__init__(model, model_params, random_state)
        self.base_model = model

    def _fit(self, X, y):
        model = self.base_model(
            n_steps=self.n_steps_,
            n_features=self.n_features_,
            n_classes=self.n_classes_,
            **self.model_params,
        )

        # Match the "train-loss-only" setup: validate on the same training set and
        # let the model select the checkpoint by the training loss trajectory.
        train_set = to_pypots(X, y)
        model.fit(train_set=train_set, val_set=train_set)
        self.model = model

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("The train-loss-only model has not been trained. Call fit() first.")
        return self.model.predict(to_pypots(X))["classification_proba"]


mascit_train_loss_only_pipeline = MASCITTrainLossOnlyWrapper(
    model=MASCIT,
    model_params={
        "n_layers": 1,
        "n_heads": 4,
        "d_model": 128,
        "d_state": 8,
        "d_conv": 4,
        "expand": 2,
        "dropout": 0.1,
        "n_kernels": 3,
        "projection_type": "gating",
        "tv_dt": False,
        "tv_B": False,
        "tv_C": False,
        "use_D": False,
        "batch_size": 16,
        "epochs": 100,
        "patience": 10,
        "num_workers": 0,
        "device": None,
        "model_saving_strategy": "best",
    },
)
"""This pipeline applies MASCIT with checkpoints selected by training loss only."""
