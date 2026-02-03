import numpy as np
from RCWA_project.base import  intermediaire, interface, c_down, c_up, cascade, layer


def local_f_prime(a, b, eta, x):
    """
    compute f' on a given segmen
    """
    mask = a <= x < b
    f_prime = 1 - eta * np.cos(2 * np.pi * (x - a) / (b - a))
    f_prime = f_prime * mask
    return f_prime


def f_prime(s, x):
    """
        compute f' (the stretching function) along a given direction
    """
    f = np.zeros_like(x)
    for i in range(len(s.ox) - 1):
        a, b = s.ox[i], s.ox[i + 1]
        f += local_f_prime(a, b, s.eta, x)
    return f


def HErmes(T, U, V, P, Amp, ny, h, a0):
    """
        Hopefully, helps compute the fields
        TODO: move this to 3D, or 2D slice (ouch)
    """
    n = int(np.shape(T)[0] / 2)
    nmod = int((n - 1) / 2)
    nx = n
    X = np.matmul(intermediaire(T, cascade(layer(V, h), U)), Amp.reshape(Amp.size, 1))
    D = X[0:n]
    X = np.matmul(intermediaire(cascade(T, layer(V, h)), U), Amp.reshape(Amp.size, 1))
    E = X[n : 2 * n]

    M = np.zeros((ny, nx - 1), dtype=complex)
    for k in range(ny):
        y = h / ny * (k + 1)
        Fourier = np.matmul(
            P,
            np.matmul(np.diag(np.exp(1j * V * y)), D)
            + np.matmul(np.diag(np.exp(1j * V * (h - y))), E),
        )
        MM = np.fft.ifftshift(Fourier[0 : len(Fourier) - 1])
        M[k, :] = MM.reshape(len(MM))
        
    M = np.conj(np.fft.ifft(np.conj(M).T, axis=0)).T * n
    x, y = np.meshgrid(np.linspace(0, 1, nx - 1), np.linspace(0, 1, ny))
    M = M * np.exp(1j * a0 * x)
    return M


def Field_2D(Ps, Vs, thickness, wavelength, angle, polarization, n_mod, period):
    """
        Hopefully, computing fields
        QO:
            - in 2D, we normalise wrt the period. Can we not do this in 3D?
                Fix idea: if we only ever compute 2D maps (slices), then the period makes sens again
    """
    
    # Normalisation
    wavelength_norm = wavelength / period

    thickness = [t / period for t in thickness]

    k0 = 2 * np.pi / wavelength_norm
    a0 = k0 * np.sin(angle)

    n_mod_total = 2 * n_mod + 1

    n_layers = thickness.size

    # matrice neutre pour l'opération de cascadage
    S11 = np.zeros((n_mod_total, n_mod_total))
    S12 = np.eye(n_mod_total)
    S1 = np.append(S11, S12, axis=0)
    S2 = np.append(S12, S11, axis=0)
    S0 = np.append(S1, S2, 1)

    # matrices d'interface
    B = []
    for k in range(n_layers - 1):  # car nc - 1 interfaces dans la structure
        a = np.array(Ps[k])
        b = np.array(Ps[k + 1])
        c = interface(a, b)
        c = c.tolist()
        B.append(c)

    S = []
    S0 = S0.tolist()
    S.append(S0)

    # Matrices montantes
    for k in range(n_layers - 1):
        a = np.array(S[k])
        b = c_up(np.array(B[k]), np.array(Vs[k]), thickness[k])
        S_new = cascade(a, b)
        S.append(S_new.tolist())

    a = np.array(S[n_layers - 1])
    b = np.array(Vs[n_layers - 1])
    c = c_down(a, b, thickness[n_layers - 1])
    S.append(c.tolist())

    # Matrices descendantes
    Q = []
    Q.append(S0)

    for k in range(n_layers - 1):
        a = np.array(B[n_layers - k - 2])
        b = np.array(Vs[n_layers - (k + 1)])
        c = thickness[n_layers - (k + 1)]
        d = np.array(Q[k])
        Q_new = cascade(c_down(a, b, c), d)
        Q.append(Q_new.tolist())

    a = np.array(Q[k])
    b = np.array(Vs[0])
    c = c_up(a, b, thickness[n_layers - (k + 1)])
    Q.append(c.tolist())

    stretch = period / (2 * n_mod + 1)

    exc = np.zeros(2 * n_mod_total) # excitation
    # Eclairage par au dessus, onde plane
    # exc[n_mod] = 1
    # eclairage par en dessous, onde plane
    # exc[n_mod_total + n_mod] = 1
    # eclairage par en dessous, guide d'onde (le mode avec la plus grande partie réelle)
    # position = np.argmax(np.real(Vdown))

    ny = np.floor(thickness * period / stretch)

    M = HErmes(
        np.array(S[0]),
        np.array(Q[n_layers - 0 - 1]),
        np.array(Vs[0]),
        np.array(Vs[0])[0:n_mod_total, 0:n_mod_total],
        exc,
        int(ny[0]),
        thickness[0],
        a0,
    )

    for j in np.arange(1, n_layers):
        M_new = HErmes(
            np.array(S[j]),
            np.array(Q[n_layers - j - 1]),
            np.array(Vs[j]),
            np.array(Ps[j])[0:n_mod_total, 0:n_mod_total],
            exc,
            int(ny[j]),
            thickness[j],
            a0,
        )
        M = np.append(M, M_new, 0)

    Mfield = np.abs(M) ** 2
    return Mfield
