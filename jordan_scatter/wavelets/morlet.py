import numpy as np
from scipy.fft import fft2, ifft2
import torch


def filter_bank(J, L, N, normalize=True):
    """
        Builds in Fourier the Morlet filters 
        Parameters
        ----------
        N : int
            spatial support of the input
        J : int
            logscale of the scattering
        L : int, optional
            number of angles used for the wavelet transform
        Notes
        -----
        The design of the filters is optimized for the value L = 8.
    """
    M = N
    hatpsi = torch.zeros(J,L,M,N) # store oriented filters
    for j in range(J):
        for theta in range(L):
            psi_signal = morlet_2d(M, N, 
                0.8 * 2**j, # sigma
                (int(L-L/2-1)-theta) * 2 * np.pi / L, # theta
                3.0 / 4.0 * np.pi /2**j) # xi
            psi_signal_fourier = np.real(fft2(psi_signal))
            # drop the imaginary part, it is zero anyway
            hatpsi[j, theta] = torch.from_numpy(psi_signal_fourier).to(dtype=torch.float32)
    phi_signal = gabor_2d(M, N, 0.8 * 2**(J-1), 0, 0)
    phi_signal_fourier = np.real(fft2(phi_signal))
    # drop the imaginary part, it is zero anyway
    hatphi = torch.from_numpy(phi_signal_fourier).to(dtype=torch.float32)

    if normalize:
        # normalized, Parseval Frame
        hatphi_square = hatphi**2
        hatpsi_square = hatpsi**2
        unity = hatpsi_square.sum(dim=(0,1)) + hatphi_square
        hatphi /= torch.sqrt(unity)
        hatpsi /= torch.sqrt(unity)

    return {"lp": hatphi, "hp":hatpsi}


def morlet_2d(M, N, sigma, theta, xi, slant=1.0, offset=0):
    """
        Computes a 2D Morlet filter.
        A Morlet filter is the sum of a Gabor filter and a low-pass filter
        to ensure that the sum has exactly zero mean in the temporal domain.
        It is defined by the following formula in space:
        psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
        where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
        the cancelling parameter.

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts

        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset)
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset)
    K = np.sum(wv) / np.sum(wv_modulus)

    mor = wv - K * wv_modulus
    return mor


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0):
    """
        Computes a 2D Gabor filter.
        A Gabor filter is defined by the following formula in space:
        psi(u) = g_{sigma}(u) e^(i xi^T u)
        where g_{sigma} is a Gaussian envelope and xi is a frequency.

        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts

        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab += np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab /= norm_factor

    return gab
