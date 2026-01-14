import numpy as np
from scipy.special import erf


# Simply a few permittivity functions, for ease of use


def epsAubb(lam):

    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam

    f0 = 0.770
    Gamma0 = 0.050
    omega_p = 9.03
    f = np.array([0.054, 0.050, 0.312, 0.719, 1.648])
    Gamma = np.array([0.074, 0.035, 0.083, 0.125, 0.179])
    omega = np.array([0.218, 2.885, 4.069, 6.137, 27.97])
    sigma = np.array([0.742, 0.349, 0.830, 1.246, 1.795])

    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    # Conversion

    epsilon = (
        1
        - omega_p**2 * f0 / (w * (w + 1.0j * Gamma0))
        + np.sum(
            1.0j
            * np.sqrt(np.pi)
            * f
            * omega_p**2
            / (2 * np.sqrt(2) * a * sigma)
            * (faddeeva(x, 64) + faddeeva(y, 64))
        )
    )
    return epsilon


def epsAgbb(lam):
    "Permet de caluler la permittivité de l'argent en une longueur d'onde lam donnée"
    w = 6.62606957e-25 * 299792458 / 1.602176565e-19 / lam
    f0 = 0.821
    Gamma0 = 0.049
    omega_p = 9.01
    f = np.array([0.050, 0.133, 0.051, 0.467, 4.000])
    Gamma = np.array([0.189, 0.067, 0.019, 0.117, 0.052])
    omega = np.array([2.025, 5.185, 4.343, 9.809, 18.56])
    sigma = np.array([1.894, 0.665, 0.189, 1.170, 0.516])
    a = np.sqrt(w * (w + 1.0j * Gamma))
    a = a * np.sign(np.real(a))
    x = (a - omega) / (np.sqrt(2) * sigma)
    y = (a + omega) / (np.sqrt(2) * sigma)
    # Conversion
    aha = (
        1.0j
        * np.sqrt(np.pi)
        * f
        * omega_p**2
        / (2 * np.sqrt(2) * a * sigma)
        * (faddeeva(x, 64) + faddeeva(y, 64))
    )
    epsilon = 1 - omega_p**2 * f0 / (w * (w + 1.0j * Gamma0)) + np.sum(aha)
    return epsilon


def faddeeva(z, N):
    "Bidouille les signes et les parties réelles et imaginaires d'un nombre complexe --> à creuser"
    w = np.zeros(z.size, dtype=complex)

    idx = np.real(z) == 0
    w[idx] = np.exp(np.abs(-z[idx] ** 2)) * (1 - erf(np.imag(z[idx])))
    idx = np.invert(idx)
    idx1 = idx + np.imag(z) < 0

    z[idx1] = np.conj(z[idx1])

    M = 2 * N
    M2 = 2 * M
    k = np.arange(-M + 1, M)
    L = np.sqrt(N / np.sqrt(2))

    theta = k * np.pi / M
    t = L * np.tan(theta / 2)
    f = np.exp(-(t**2)) * (L**2 + t**2)
    f = np.append(0, f)
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / M2
    a = np.flipud(a[1 : N + 1])

    Z = (L + 1.0j * z[idx]) / (L - 1.0j * z[idx])
    p = np.polyval(a, Z)
    w[idx] = 2 * p / (L - 1.0j * z[idx]) ** 2 + (1 / np.sqrt(np.pi)) / (
        L - 1.0j * z[idx]
    )
    w[idx1] = np.conj(2 * np.exp(-z[idx1] ** 2) - w[idx1])
    return w
