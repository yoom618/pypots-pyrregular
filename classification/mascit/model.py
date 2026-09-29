"""
The implementation of MASCIT for the partially-observed time-series classification task.

"""

from typing import Optional, Union

import torch

from .core import _MASCIT
from pypots.classification.base import BaseNNClassifier
from pypots.nn.modules.loss import Criterion, CrossEntropy
from pypots.optim.adam import Adam
from pypots.optim.base import Optimizer


class MASCIT(BaseNNClassifier):
    """The PyTorch implementation of the MASCIT classification model.

    Notes
    -----
    This implementation follows the single-layer MASCIT design for sequence classification and
    is adapted to PyPOTS by explicitly consuming ``missing_mask`` together with ``X``.

    The default arguments are set to a gating-centered profile aligned with the
    final MASCIT experiment scripts where possible (e.g. ``projection_type``,
    ``expand``, ``d_conv``, ``dropout``, ``batch_size``, ``epochs``, ``patience``).
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_classes: int,
        d_model: int = 128,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 1,
        dropout: float = 0.1,
        n_kernels: int = 3,
        projection_type: str = "gating",
        n_heads: int = 8,
        tv_dt: bool = False,
        tv_B: bool = False,
        tv_C: bool = False,
        use_D: bool = False,
        batch_size: int = 16,
        epochs: int = 100,
        patience: Optional[int] = 10,
        training_loss: Union[Criterion, type] = CrossEntropy,
        validation_metric: Union[Criterion, type] = CrossEntropy,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            n_classes=n_classes,
            training_loss=training_loss,
            validation_metric=validation_metric,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout
        self.n_kernels = n_kernels
        self.projection_type = projection_type
        self.n_heads = n_heads
        self.tv_dt = tv_dt
        self.tv_B = tv_B
        self.tv_C = tv_C
        self.use_D = use_D

        self.model = _MASCIT(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_classes=self.n_classes,
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dropout=self.dropout,
            n_kernels=self.n_kernels,
            projection_type=self.projection_type,
            n_heads=self.n_heads,
            tv_dt=self.tv_dt,
            tv_B=self.tv_B,
            tv_C=self.tv_C,
            use_D=self.use_D,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )
        self._send_model_to_given_device()
        self._print_model_size()

        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())
