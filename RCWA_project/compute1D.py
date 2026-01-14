import numpy as np
import scipy.linalg as lin


def f(s, x):
    """
    Stretching function along x
    """
    j = np.argmax((s.nx - x) > 0) - 1

    new_diff = s.nx[j + 1] - s.nx[j]
    new_k = 2 * np.pi / new_diff
    old_diff = s.ox[j + 1] - s.ox[j]

    val = s.ox[j] + old_diff / new_diff * (
        x - s.nx[j] - s.eta * np.sin(new_k * (x - s.nx[j])) / new_k
    )
    return val


def step(eps_1, eps_2, beg, end, n):
    """
    Computes the Fourier series for a piecewise function having the value
    eps_2 between beg and end, and the value eps_1 otherwise.
    The period is supposed to be equal to 1.
    Then returns the toeplitz matrix generated using the Fourier series.
    """

    w = end - beg

    l = np.zeros(n, dtype=np.complex128)
    m = np.zeros(n, dtype=np.complex128)

    tmp = (
        np.exp(-2 * 1j * np.pi * (beg + w / 2) * np.arange(0, n))
        * np.sinc(w * np.arange(0, n))
        * w
    )
    l = np.conj(tmp) * (eps_2 - eps_1)
    m = tmp * (eps_2 - eps_1)
    l[0] = l[0] + eps_1
    m[0] = l[0]
    T = lin.toeplitz(l, m)

    return T


def structured(epsilons, interfaces, k0, kx, pol, Mm, verbose=False):
    """
    Computes the modes and eigenvalues in a structured layer
    """
    N = 2 * Mm + 1
    eps_base = epsilons[0]

    # The Toeplitz matrix for the direct decomposition
    T_direct = eps_base * np.eye(N, N)
    if pol == 1:
        # The Toeplitz matrix for the inverse decomposition
        T_inv = 1 / eps_base * np.eye(N, N)

    for k in range(1, len(interfaces)-1):
        # Build the Toeplitz matrices describing the system
        eps_change = epsilons[k]
        beg = interfaces[k]
        end = interfaces[k+1]

        T_direct = T_direct + step(0, eps_change - eps_base, beg, end, N)
        if pol == 1:
            T_inv = T_inv + step(0, 1 / eps_change - 1 / eps_base, beg, end, N)

    # The kx matrix
    alpha = np.diag(kx + 2 * np.pi * np.arange(-Mm, Mm + 1)) + 0j

    if pol == 0:
        # TE
        M = alpha**2 - k0**2 * T_direct
        # M represents the eigen value problem given by Maxwell's equations
        V, E = np.linalg.eig(M)

        V = np.sqrt(-V + 0j)
        # Keep only significant imaginary parts of eigen values
        keep = np.abs(np.imag(V)) > 1e-10
        V = np.real(V) + 1.0j * (np.imag(V) * keep)
        # Keeping positive imaginary part solutions
        neg_val = np.imag(V) < 0
        V = V * (1 - 2 * neg_val) 

        P = np.block([[E],
                      [E @ np.diag(V)]])

    else:
        # TM
        T = np.linalg.inv(T_inv)
        M = (T @ alpha @ np.linalg.inv(T_direct) @ alpha - k0**2 * T)
        # M represents the eigen value problem given by Maxwell's equations

        V, E = np.linalg.eig(M)
        V = np.sqrt(-V + 0j)

        # Keep only significant imaginary parts of eigen values
        keep = np.abs(np.imag(V)) > 1e-10
        V = np.real(V) + 1.0j * (np.imag(V) * keep)
        # Keeping positive imaginary part solutions
        neg_val = np.imag(V) < 0
        V = V * (1 - 2 * neg_val) 

        P = np.block([[E],
                      [T_inv @ E @ np.diag(V)]])

    return P, V


def homogeneous(epsilon, k0, kx, pol, Mm, verbose=False):
    """
    Computing modes and eigenvalues in a homogeneous layer, in 1D, is straightforward
    """
    V = np.sqrt(epsilon * k0**2 - (kx + 2 * np.pi * np.arange(-Mm, Mm + 1)) ** 2 + 0j)

    # Keep only significant imaginary parts of eigen values
    keep = np.abs(np.imag(V)) > 1e-10
    V = np.real(V) + 1.0j * (np.imag(V) * keep)
    # Keeping positive imaginary part solutions
    neg_val = np.imag(V) < 0
    V = V * (1 - 2 * neg_val) 

    P = np.block([[np.eye(2*Mm + 1)],
                  [np.diag(V * (pol / epsilon + (1 - pol)))]])

    return P, V

