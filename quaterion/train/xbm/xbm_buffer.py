from typing import Tuple

import torch
from torch import Tensor, LongTensor

from .xbm_config import XbmConfig, XbmDevice


class XbmBuffer:
    """A buffer implementation to hold recent N embeddings and target values.

    Inspired by https://github.com/msight-tech/research-xbm/blob/master/ret_benchmark/modeling/xbm.py

    Args:
        config: Config class to configure XBM settings.
    """

    def __init__(self, config: XbmConfig):
        if config.device == XbmDevice.AUTO:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.device

        self._cfg = config

        self._embeddings = torch.zeros(
            self._cfg.buffer_size, self._cfg.embedding_size
        ).to(device)
        self._targets = torch.zeros(self._cfg.buffer_size, torch.long).to(device)
        self._pointer = 0
        self._is_full = False

    @property
    def is_full(self) -> bool:
        return self._is_full

    def get(self) -> Tuple[Tensor, LongTensor]:
        if self.is_full:
            return self._embeddings, self._targets
        else:
            return self._embeddings[: self._pointer], self._targets[: self._pointer]

    def queue(self, embeddings: Tensor, targets: LongTensor) -> None:
        """Queue batch embeddings and targets in the buffer.

        Args:
            embeddings: Output embeddings in the batch.
            targets: Target values in the batch.
        """
        batch_size = len(targets)

        if self._pointer + batch_size > self._cfg.buffer_size:
            self._embeddings[-batch_size:] = embeddings
            self._targets[-batch_size:] = targets
            self._pointer = 0
            if not self.is_full:
                self._is_full = True

        else:
            self._embeddings[self._pointer : self._pointer + batch_size] = embeddings
            self._targets[self._pointer : self._pointer + batch_size] = targets
            self._pointer += batch_size