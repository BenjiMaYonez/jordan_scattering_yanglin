# Key Components
- `jordan_scatter`: subpackage for pytorch model definition
- `configs`: subpackage for configuration of hyperparameter
- `scripts`: command-line entry point
- `experiments`: saved experiments and configuration for reprodusibility
- `images`: subpackage for image utility, saved images
# Get Start

## Package Installation
Before running, install the package in your python environment `pip install -e .`

## Running file
Sanity Check, lossless inversion: `python scripts/inverse.py --config configs/full_inverse.yaml`

Invertible: `python scripts/inverse.py --config configs/inverse.yaml`

## Apple Silicon GPU support

- Use Python 3.10+ and install a PyTorch build with Metal backend (e.g. `pip3 install --upgrade torch torchvision torchaudio`).
- On macOS 12.3+ with an M-series chip, the scripts will automatically prefer the GPU; `scripts/inverse.py` now falls back to CUDA, then MPS, then CPU.
- The runner will automatically retry on CPU if the GPU backend reports an out-of-memory or Metal command-buffer fault, so long jobs still finish.
- You can limit the largest tensor kept on the accelerator by exporting `JORDAN_DEVICE_TENSOR_LIMIT_GB` (defaults to 8 GiB for CUDA/MPS); once the estimate is larger than this, the model transparently moves the remaining layers to CPU.
- Optional optimizations (disabled by default) can be toggled in each YAML config under `optimizations`:
  - `mixed_precision: true` executes FFT/Jordan blocks in float16 on CUDA/MPS while leaving filters in float32.
  - `force_float16: true` converts the entire model and inputs to float16 on accelerators, with automatic float32 fallback on CPU runs.
  - `disk_cache: true` persists layer outputs to disk between forward and inverse passes so large runs avoid keeping everything in GPU RAM.
- Export `PYTORCH_ENABLE_MPS_FALLBACK=1` to allow unsupported ops (such as some FFT kernels) to transparently run on CPU while keeping everything else on the GPU: `export PYTORCH_ENABLE_MPS_FALLBACK=1`.
- You can verify the device that will be used with:
  ```python
  python3 - <<'PY'
  import torch
  if torch.cuda.is_available():
      print('CUDA available:', torch.cuda.get_device_name())
  elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
      print('MPS available')
  else:
      print('Falling back to CPU')
  PY
  ```
