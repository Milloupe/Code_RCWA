import numpy as np
import scipy.linalg as lin
import RCWA_project.base as base


def compute_PV(struct, wavelength, interfaces, k0, kx, pol, Mm, eta=0):
    """
    Docstring for compute_PV
    
    :param struct: Description
    :param wavelength: Description
    :param interfaces: Description
    :param k0: Description
    :param kx: Description
    :param pol: Description
    :param Mm: Description
    """
    Ps = []
    Vs = []
    nb_layer = len(struct.layers)

    if eta:
        for ilayer in range(nb_layer):
            layer = struct.layers[ilayer]
            epsilons = [mat.get_permittivity(wavelength) for mat in layer]
            mus = [mat.get_permeability(wavelength) for mat in layer]
            eig_vec, eig_val = stretched(epsilons, mus, interfaces, k0, kx, pol, eta, Mm)
            Ps.append(eig_vec)
            Vs.append(eig_val)
    else:

        for ilayer in range(nb_layer):
            layer = struct.layers[ilayer]
            homo = struct.homo_layer[ilayer]
            if homo:
                epsilon = layer[0].get_permittivity(wavelength)
                eig_vec, eig_val = homogeneous(epsilon, k0, kx, pol, Mm)
            else:
                epsilons = [mat.get_permittivity(wavelength) for mat in layer]
                eig_vec, eig_val = structured(epsilons, interfaces, k0, kx, pol, Mm) 

            Ps.append(eig_vec)
            Vs.append(eig_val)
    return Ps, Vs

def f_times_perm(perm, interfaces, eta, modes):
    """
    Computing the Toeplitz matrix for perm * f
    """
    m = 2 * modes + 1

    v = np.zeros(2*m+1, dtype=complex)
    for j in range(len(interfaces)-1):
        a = interfaces[j]
        b = interfaces[j+1]
        tfx = base.tfd(a, b, eta, m)
        v = v + perm[j] * tfx # * (1 + 1.0j * s.pmlx[j])  # f * perm
    v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
    T = base.toep(v)
    return T


def f_over_perm(perm, interfaces, eta, modes):
    """
    Computing the Toeplitz matrix for f / perm
    """
    m = 2 * modes + 1

    v = np.zeros(2*m+1, dtype=complex)
    for j in range(len(interfaces)-1):
        a = interfaces[j]
        b = interfaces[j+1]
        tfx = base.tfd(a, b, eta, m)
        v = v + 1 / perm[j] * tfx # * (1 + 1.0j * s.pmlx[j])  # f / perm
    v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
    T = base.toep(v)
    return T


def stretched(epsilons, mus, interfaces, k0, kx, pol, eta, Mm):
    """
    """

    # The kx matrix
    alpha = np.diag(kx + 2 * np.pi * np.arange(-Mm, Mm + 1)) + 0j

    if pol == 0:
        # TE
        perm1 = mus
        perm2 = epsilons
    else:
        # TM
        perm1 = epsilons
        perm2 = mus

    T_f_over_mu = f_over_perm(perm1, interfaces, eta, Mm)
    T_f_mu = np.linalg.inv(f_times_perm(perm1, interfaces, eta, Mm))
    T_f_eps = f_times_perm(perm2, interfaces, eta, Mm)

    L =  alpha @ T_f_mu @ alpha - k0**2 * T_f_eps
    M = np.linalg.inv(T_f_over_mu) @ L
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
                    [L @ E @ np.diag(1/V)]])

    return P, V

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


def step(eps_diff, beg, end, n):
    """
    Computes the Fourier series for a piecewise function having the value
    eps_2 between beg and end, and the value eps_1 otherwise.
    The period is supposed to be equal to 1.
    Then returns the toeplitz matrix generated using the Fourier series.
    """

    w = end - beg
    cen = (end + beg) / 2

    l = np.zeros(2*n+1, dtype=np.complex128)
    m = np.zeros(2*n+1, dtype=np.complex128)

    ks = np.arange(0, 2*n+1)

    tmp = (
        np.exp(-2j * np.pi * cen * ks)
        * np.sinc(w * ks)
        * w
    )
    l = np.conj(tmp) * (eps_diff)
    m = tmp * (eps_diff)
    l[0] = l[0]
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
        if (eps_change != eps_base):
            beg = interfaces[k]
            end = interfaces[k+1]

            T_direct = T_direct + step(eps_change - eps_base, beg, end, Mm)
            if pol == 1:
                T_inv = T_inv + step(1 / eps_change - 1 / eps_base, beg, end, Mm)

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
        M = T @ alpha @ np.linalg.inv(T_direct) @ alpha - k0**2 * T
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

