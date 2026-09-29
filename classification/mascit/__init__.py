"""Minimal MASCIT package required by the single-dataset runner."""

from .model import MASCIT
from .mascit_train_loss_only import MASCITTrainLossOnlyWrapper

__all__ = ["MASCIT", "MASCITTrainLossOnlyWrapper"]
