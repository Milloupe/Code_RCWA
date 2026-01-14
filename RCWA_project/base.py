import numpy as np
import scipy.linalg as lin


def cascade(U, V):
    """Cascading of two scattering matrices U and V.
    Since U and V are scattering matrices, it is expected that they are square
    and have the same dimensions which are necessarily EVEN.
    HDR 1.43
    """
    n = int(U.shape[0] / 2)
    U00 = U[0:n, 0:n]
    U01 = U[0:n, n : 2 * n]
    U10 = U[n : 2 * n, 0:n]
    U11 = U[n : 2 * n, n : 2 * n]

    V00 = V[0:n, 0:n]
    V01 = V[0:n, n : 2 * n]
    V10 = V[n : 2 * n, 0:n]
    V11 = V[n : 2 * n, n : 2 * n]

    J = np.linalg.inv(np.eye(n) - V00 @ U11)
    K = np.linalg.inv(np.eye(n) - U11 @ V00)

    S = np.block(
        [
            [U00 + U01 @ J @ V00 @ U10, U01 @ J @ V01],
            [V10 @ K @ U10, V11 + V10 @ K @ U11 @ V01],
        ]
    )
    return S


def c_down(A, V, h):
    """Directly cascading any scattering matrix A (square and with even
    dimensions) with the scattering matrix of a layer of thickness h in which
    the wavevectors are given by V. Since the layer matrix is
    essentially empty, the cascading is much quicker if this is taken
    into account.
    """
    n = int(A.shape[0] / 2)
    D = np.diag(np.exp(1.0j * V * h))
    S = np.block(
        [
            [A[0:n, 0:n], A[0:n, n : 2 * n] @ D],
            [D @ A[n : 2 * n, 0:n], D @ A[n : 2 * n, n : 2 * n] @ D],
        ]
    )
    return S


def c_up(A, V, h):
    """
    Docstring for c_up

    :param A: Description
    :param V: Description
    :param h: Description
    """
    n = int(A[0].size / 2)
    D = np.diag(np.exp(1.0j * V * h))
    S = np.block(
        [
            [D @ A[0:n, 0:n] @ D, D @ A[0:n, n : 2 * n]],
            [A[n : 2 * n, 0:n] @ D, A[n : 2 * n, n : 2 * n]],
        ]
    )
    return S


def intermediaire(T, U):
    """
    Docstring for intermediaire

    :param T: Description
    :param U: Description
    """
    n = T.shape[0] // 2
    H = np.linalg.inv(np.eye(n) - U[0:n, 0:n] @ T[n : 2 * n, n : 2 * n])
    K = np.linalg.inv(np.eye(n) - T[n : 2 * n, n : 2 * n] @ U[0:n, 0:n])
    a = K @ T[n : 2 * n, 0:n]
    b = K @ T[n : 2 * n, n : 2 * n] @ U[0:n, n : 2 * n]
    c = H @ U[0:n, 0:n] @ T[n : 2 * n, 0:n]
    d = H @ U[0:n, n : 2 * n]
    S = np.block([[a, b], [c, d]])
    return S


def interface(P, Q):
    """
    Computation of the scattering matrix of an interface, P and Q being the
    matrices given for each layer by homogene, or structured.
    P[:n,   i] is the x component of the mode i
    P[n:2:, i] is the y component of the mode i
    The S matrix computed is simply the solution to the continuity relations over the
    EM field at the interface
    """
    n = int(P.shape[1])
    A = np.block([[P[0:n, 0:n], -Q[0:n, 0:n]],
                  [P[n : 2 * n, 0:n], Q[n : 2 * n, 0:n]]])
    B = np.block([[-P[0:n, 0:n], Q[0:n, 0:n]],
                  [P[n : 2 * n, 0:n], Q[n : 2 * n, 0:n]]])
    S = np.linalg.inv(A) @ B
    return S


# def interface_3D(P, Q, V1, V2, L1, L2):
#     """
#         Computation of the scattering matrix of an interface, P and Q being the
#         matrices given for each layer by homogene, reseau or creneau.
#         P[:n,   i] is the x component of the mode i
#         P[n:2:, i] is the y component of the mode i
#         The S matrix computed is simply the solution to the continuity relations over the
#         EM field at the interface
#     """
#     #TODO: update all this to comply with HDR 1.80
#     n = int(P.shape[1])
#     A = np.block([[P[0:n, 0:n], -Q[0:n, 0:n]],
#                   [P[n:2*n, 0:n], Q[n:2*n, 0:n]]])
#     B = np.block([[-P[0:n, 0:n], Q[0:n, 0:n]],
#                   [P[n:2*n, 0:n], Q[n:2*n, 0:n]]])
#     S = np.linalg.inv(A) @ B
#     return S


def genere(ox, nx, eta, n):
    """
    Computes the fourier transform of coordinates, with stretching,
    for all interface coordinates
    """
    fp = []
    for i in range(len(ox) - 1):
        fp.append(tfd(ox[i], ox[i + 1], nx[i], nx[i + 1], eta, nx[-1], n).T)
    return np.array(fp).T  # TODO: check whether transposing is necessary


def tfd(old_a, old_b, new_a, new_b, eta, d, N):
    """
    Computing fourier transform of coordinates with stretching
    """
    pi = np.pi
    fft = np.zeros(2 * N + 1, dtype=complex)
    old_ba = old_b - old_a
    new_ba = new_b - new_a

    # Homogeneous layer, only one zone
    # TODO: this doesn't really work, because the period is a direct multiple of the zone size
    if old_a == new_a == 0 and old_b == new_b == d:
        for i_mod in range(-N, N + 1):
            if i_mod == 0:
                fft[N] = 1
            elif (i_mod == 1) or (i_mod == -1):
                fft[i_mod + N] = -eta / 2
            else:
                fft[i_mod + N] = (
                    -1
                    / (2j * np.pi)
                    * (np.exp(-2j * np.pi * i_mod) - 1)
                    * (1 / i_mod + eta * i_mod / (1 - i_mod**2))
                )
    # Heterogeneous layer
    else:
        for i_mod in range(-N, N + 1):
            sinc_prefac = (
                old_ba
                * i_mod
                / d
                * np.sinc(i_mod / d * new_ba)
                * np.exp(-1.0j * np.pi * i_mod * (new_b + new_a) / d)
            )

            n_diff = i_mod * new_ba
            if i_mod == 0:
                fft[N] = old_ba / d
            elif d - n_diff == 0:
                # fft[i_mod + N] = prefac * (1/i_mod - eta/2 * new_ba/(d+n_diff)) * (exp_kb-exp_ka) - eta/2 * old_ba*np.exp(-2.0j*pi*new_a/new_ba)/d
                fft[i_mod + N] = (
                    sinc_prefac * (1 / i_mod - eta / 2 * new_ba / (d + n_diff))
                    - eta / 2 * old_ba * np.exp(-2.0j * pi * new_a / new_ba) / d
                )
            elif d + n_diff == 0:
                # fft[i_mod + N] = prefac * (1/i_mod + eta/2 * new_ba/(d-n_diff)) * (exp_kb-exp_ka) - eta/2 * old_ba*np.exp(2.0j*pi*new_a/new_ba)/d
                fft[i_mod + N] = (
                    sinc_prefac * (1 / i_mod + eta / 2 * new_ba / (d - n_diff))
                    - eta / 2 * old_ba * np.exp(2.0j * pi * new_a / new_ba) / d
                )
            else:
                # fft[i_mod + N] = prefac * (1/i_mod + eta/2 * (new_ba/(d-n_diff)-new_ba/(d+n_diff))) * (exp_kb-exp_ka)
                fft[i_mod + N] = sinc_prefac * (
                    1 / i_mod
                    + eta / 2 * (new_ba / (d - n_diff) - new_ba / (d + n_diff))
                )
    return fft


"""
fft=0.;
for n=-N:N   
  if (n==0)
    fft(N+1) = (b-a)/d;
  elseif (d-n*(b1-a1)==0)
    fft(n+N+1)=-1./(2*i*pi)*(b-a)/(b1-a1)*(1/n-eta/2*(b1-a1)/(d+n*(b1-a1)))*(exp(-2*i*pi*b1*n/d)-exp(-2*i*pi*a1*n/d))-eta/2*(b-a)*exp(-2*i*pi*a1/(b1-a1))/d;
  elseif (d+n*(b1-a1)==0)
    fft(n+N+1)=-1./(2*i*pi)*(b-a)/(b1-a1)*(1/n+eta/2*(b1-a1)/(d-n*(b1-a1)))*(exp(-2*i*pi*b1*n/d)-exp(-2*i*pi*a1*n/d))-eta/2*(b-a)*exp(2*i*pi*a1/(b1-a1))/d;
  else
    fft(n+N+1)=-1./(2*i*pi)*(b-a)/(b1-a1)*(1/n+eta/2*((b1-a1)/(d-n*(b1-a1))-(b1-a1)/(d+n*(b1-a1))))*(exp(-2*i*pi*b1*n/d)-exp(-2*i*pi*a1*n/d));
  endif
endfor


"""


def toep(v):
    """
    Computing Toeplitz matrix
    """
    n = (len(v) - 1) // 2
    a = v[n:0:-1]
    b = v[n : 2 * n]
    T = lin.toeplitz(b, a)
    return T
