import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CachedTensorRef:
    cache_path: str

    def load(self, device: torch.device) -> torch.Tensor:
        return torch.load(self.cache_path, map_location=device)


class LayerDiskCache:
    """Stores large tensors on disk to reduce accelerator memory pressure."""

    def __init__(self, root_dir: Optional[str] = None, prefix: str = "layer_cache"):
        self._root_dir = root_dir
        self._prefix = prefix
        self._tmpdir: Optional[str] = None
        self._refs: list[CachedTensorRef] = []

    @property
    def root(self) -> str:
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix=f"{self._prefix}_", dir=self._root_dir)
        return self._tmpdir

    def save(self, key: str, tensor: torch.Tensor) -> CachedTensorRef:
        path = os.path.join(self.root, f"{key}.pt")
        torch.save(tensor, path)
        ref = CachedTensorRef(cache_path=path)
        self._refs.append(ref)
        return ref

    def cleanup(self) -> None:
        if self._tmpdir is None:
            return
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._refs.clear()
        self._tmpdir = None
