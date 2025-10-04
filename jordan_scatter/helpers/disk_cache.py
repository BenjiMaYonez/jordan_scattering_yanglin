import os
import tempfile
from typing import Optional

import torch


class LayerDiskCache:
    """Stores large tensors on disk to reduce accelerator memory pressure."""

    def __init__(self, root_dir: Optional[str] = None, prefix: str = "layer_cache"):
        self._root_dir = root_dir
        self._prefix = prefix
        self._tmpdir: Optional[str] = None
        self._handles = []

    @property
    def root(self) -> str:
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix=f"{self._prefix}_", dir=self._root_dir)
        return self._tmpdir

    def save(self, key: str, tensor: torch.Tensor) -> dict:
        path = os.path.join(self.root, f"{key}.pt")
        torch.save(tensor, path)
        self._handles.append(path)
        return {"cache_path": path, "shape": tuple(tensor.shape)}

    @staticmethod
    def load(ref: dict, device: torch.device) -> torch.Tensor:
        path = ref["cache_path"]
        tensor = torch.load(path, map_location=device)
        return tensor

    def cleanup(self) -> None:
        if not self._handles:
            return
        for path in self._handles:
            try:
                os.remove(path)
            except OSError:
                pass
        if self._tmpdir is not None:
            try:
                os.rmdir(self._tmpdir)
            except OSError:
                pass
        self._handles.clear()
        self._tmpdir = None

