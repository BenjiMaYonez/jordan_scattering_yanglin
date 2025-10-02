import numpy as np
import torch

_TAU = np.float32(32 * np.finfo(np.float32).eps)
_EPS = np.float32(1e-12)


def _fftfreq2_rad(M: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return radial magnitude and angle grids for a 2-D Fourier domain."""
    wx = 2 * np.pi * np.fft.fftfreq(M, d=1.0)
    wy = 2 * np.pi * np.fft.fftfreq(N, d=1.0)
    WY, WX = np.meshgrid(wy, wx, indexing="ij")
    R = np.sqrt(WX ** 2 + WY ** 2).astype(np.float32)
    Theta = np.arctan2(WY, WX).astype(np.float32)
    return R, Theta


def _radial_lowpass(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    L = np.zeros_like(u, dtype=np.float32)
    L[u <= np.float32(0.5)] = np.float32(1.0)
    mask = (u > np.float32(0.5)) & (u < np.float32(1.0) + _TAU)
    if np.any(mask):
        x = 2.0 * np.maximum(u[mask], np.float32(1e-12))
        x = np.clip(x, np.float32(1.0), np.float32(2.0 - _TAU))
        t = np.log2(x)
        L[mask] = np.cos(0.5 * np.pi * t).astype(np.float32)
    return L


def _radial_bandpass(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    B = np.zeros_like(u, dtype=np.float32)
    mask = (u > np.float32(0.5) - _TAU) & (u < np.float32(2.0) + _TAU)
    if np.any(mask):
        z = np.maximum(u[mask], np.float32(1e-12))
        z = np.clip(z, np.float32(0.5 + _TAU), np.float32(2.0 - _TAU))
        t = np.log2(z)
        B[mask] = np.cos(0.5 * np.pi * t).astype(np.float32)
    return B


def _wrap_period(x: np.ndarray, period: float) -> np.ndarray:
    return (x + 0.5 * period) % period - 0.5 * period


def _angular_windows_L2norm(theta: np.ndarray, L: int, eps: float = 1e-12) -> list[np.ndarray]:
    centers = [(ell * np.pi) / L for ell in range(L)]
    windows = []
    for c in centers:
        diff = _wrap_period(theta - np.float32(c), np.pi)
        u = (L / np.pi) * diff
        window = np.zeros_like(theta, dtype=np.float32)
        support = np.abs(u) <= np.float32(1.0)
        if np.any(support):
            uc = np.clip(u[support], -1.0, 1.0)
            window[support] = np.cos(0.5 * np.pi * uc).astype(np.float32)
        windows.append(window)
    stack = np.stack(windows, axis=0)
    denom = np.sqrt(np.sum(stack ** 2, axis=0) + np.float32(eps)).astype(np.float32)
    zeros = denom < np.sqrt(np.float32(eps))
    if np.any(zeros):
        stack[:, zeros] = 0.0
        stack[0, zeros] = 1.0
        denom[zeros] = 1.0
    return [(stack[k] / denom).astype(np.float32) for k in range(L)]


def filter_bank(J: int, L: int, N: int, normalize: bool = True) -> dict[str, torch.Tensor]:
    """Build a Fourier-domain Meyer wavelet filter bank.

    Args:
        J: Number of dyadic scales.
        L: Number of angular sectors.
        N: Spatial size (assumes square input).

    Returns:
        Dictionary with keys ``"lp"`` (low-pass) and ``"hp"`` (high-pass).
    """
    R, Theta = _fftfreq2_rad(N, N)
    u = R / np.pi  # normalize radius so that Nyquist along axes maps to 1

    lp = _radial_lowpass((2.0 ** (J - 1)) * u)
    hp = np.zeros((J, L, N, N), dtype=np.float32)

    angular_windows = _angular_windows_L2norm(Theta, L)

    for j in range(J):
        radial = _radial_bandpass((2.0 ** j) * u)
        for ell, gamma in enumerate(angular_windows):
            hp[j, ell] = (radial * gamma).astype(np.float32)

    if normalize:
        unity = np.sum(hp ** 2, axis=(0, 1)) + lp ** 2
        unity = np.maximum(unity, _EPS)
        sqrt_unity = np.sqrt(unity).astype(np.float32)
        hp /= sqrt_unity
        lp /= sqrt_unity

    return {
        "lp": torch.from_numpy(lp.astype(np.float32)),
        "hp": torch.from_numpy(hp.astype(np.float32)),
    }


__all__ = ["filter_bank"]
