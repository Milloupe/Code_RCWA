import numpy as np
import RCWA_project.compute1D as compute1D
import RCWA_project.compute2D as compute2D
import RCWA_project.base_functions as base

""" TODO:
    - create r, t, R, T functions for 1D, 1D angle and 2D
    - create Field functions for 1D, 1D angle and 2D
    - create wrappers that automatically detect which should be called
    (criteria: structure in y has only one zone -> 1D, no phi angle -> pure 1D)
    -> eventually, the 'structure' class will let me check the validity of the structure!
"""

def coefficient(struct, wavelength, incidence, n_mod, pmls=0, eta=0):
    """
    Docstring for coefficient
    
    :param struct: Description
    :param wavelength: Description
    :param incidence: Description
    :param n_mod: Description
    :param pml: Description
    :param eta: Description
    """
    # TODO manage edge cases, make sure the dimensions are all right,
    # maybe streamline a bit

    if struct.type == "1D" and len(incidence) == 2:
        # 1D structure
        if not(eta or pmls):
            # No stretching needed
            return coefficient_1D(struct, wavelength, incidence, n_mod)
        else:
            print("Please use a 2d structure (even pseudo-2D) when stretching",
                  ", this includes using PMLs")
    else:
        return coefficient_2D(struct, wavelength, incidence, n_mod, pmls, eta)


def coefficient_1D(struct, wavelength, incidence, n_mod, eta=0):
    """
    This function computes the reflection and transmission coefficients
    of a 1D struct when the incidence plane is the same as the struct plane.

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (float): incidence angle in radians and polarization (1 TM or 0 TE)
        n_mod (int): number of Fourier modes in the decomposition

    returns:
        r (complex): reflection coefficient, phase origin at first interface
        t (complex): transmission coefficient
        R (float): Reflectance (energy reflection)
        T (float): Transmittance (energie transmission)


    R and T are the energy coefficients (real quantities)
    """
    k0 = 2 * np.pi / wavelength * struct.period
    theta, pol = incidence  # In 1D, only one angle of incidence is necessary
    kx = -k0 * np.sin(theta)
    # Normalize everything

    # General structure parameters
    interfaces = np.array(struct.interfaces) / struct.period
    nb_layer = len(struct.thicknesses)

    Ps, Vs = compute1D.compute_PV(
        struct, wavelength, interfaces, k0, kx, pol, n_mod, eta=eta
    )

    for ilayer in range(nb_layer):
        thickness = np.array(struct.thicknesses[ilayer]) / struct.period
        # Create the S matrix of the system
        if ilayer > 0:
            S_layer = base.c_down(
                base.interface(Ps[ilayer - 1], Ps[ilayer]),
                Vs[ilayer],
                thickness,
            )

            # Once inside the system, build the complete system S matrix
            if ilayer > 1:
                S = base.cascade(S, S_layer)
            else:
                S = S_layer

    r = S[n_mod, n_mod]
    t = S[3 * n_mod + 1, n_mod]
    kz_t = np.real(Vs[-1][n_mod])
    perm_top = struct.get_perm_top(wavelength)
    perm_bot = struct.get_perm_bot(wavelength)
    R = np.abs(r**2)
    T = np.abs(t) ** 2 * kz_t / (k0 * np.cos(theta)) * (perm_top / perm_bot)

    return r, t, R, T


def coefficient_1D_angle(struct, wavelength, incidence):
    """
    This function computes the reflection and transmission coefficients
    of a 1D struct when the incidence plane is not the same as the structured plane.

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (list): incidence angles (theta, phi, pol) in radians

    returns:
        r (complex): reflection coefficient, phase origin at first interface
        t (complex): transmission coefficient
        R (float): Reflectance (energy reflection)
        T (float): Transmittance (energie transmission)


    R and T are the energy coefficients (real quantities)
    """
    # Do stuff
    return None


def coefficient_2D(struct, wavelength, incidence, n_mod, pmls, eta):
    """
    This function computes the reflection and transmission coefficients
    of a 2D structure.

    Args:
        struct (Structure): belongs to the Structure class
        wavelength (float): wavelength of the incidence light (in nm)
        incidence (list): incidence angles (theta, pol, phi) in radians

    returns:
        r (complex): reflection coefficient, phase origin at first interface
        t (complex): transmission coefficient
        R (float): Reflectance (energy reflection)
        T (float): Transmittance (energie transmission)


    R and T are the energy coefficients (real quantities)
    """
    k0 = 2 * np.pi / wavelength
    theta, pol, phi = incidence
    kx = -k0 * np.sin(theta) * np.cos(phi)
    ky = -k0 * np.sin(theta) * np.sin(phi)
    n_mod_x, n_mod_y = n_mod

    # General structure parameters
    int_x = np.array(struct.int_x)
    int_y = np.array(struct.int_y)
    nb_layer = len(struct.thicknesses)

    Ps, Vs, ext = compute2D.compute_PV(
        struct, wavelength, int_x, int_y, k0, kx, ky, n_mod, pmls, eta=eta
    )

    for ilayer in range(nb_layer):
        thickness = np.array(struct.thicknesses[ilayer])
        # Create the S matrix of the system
        if ilayer > 0:
            S_layer = base.c_down(
                base.interface(Ps[ilayer - 1], Ps[ilayer]),
                Vs[ilayer],
                thickness,
            )

            # Once inside the system, build the complete system S matrix
            if ilayer > 1:
                S = base.cascade(S, S_layer)
            else:
                S = S_layer

    # Computing the reflected and transmitted coefficients
    perm_top = struct.get_perm_top(wavelength)
    perm_bot = struct.get_perm_bot(wavelength)
    Ex = np.cos(pol) * np.cos(theta) * np.cos(phi) - np.sin(pol) * np.sin(
        phi
    )  # Ex incident
    Ey = np.cos(pol) * np.cos(theta) * np.sin(phi) + np.sin(pol) * np.cos(
        phi
    )  # Ey incident

    eps_k2 = perm_top * k0**2  # eps k^2
    d = np.sqrt(eps_k2 - kx**2 - ky**2)  # norm kz
    # e = normalisation E
    norm = (
        (eps_k2 - ky**2) * np.abs(Ex) ** 2
        + (eps_k2 - kx**2) * np.abs(Ey) ** 2
        + 2 * kx * ky * np.real(Ex * Ey)
    ) / (d)

    ext_kz, _, _, ext_pos = ext
    nb_mod = np.floor(np.real(ext_kz[0])).astype(int)
    V_inc = np.zeros(4 * (2 * n_mod_y + 1) * (2 * n_mod_x + 1), dtype=complex)
    V_inc[ext_pos[0]] = Ex / np.sqrt(norm)
    V_inc[ext_pos[nb_mod]] = Ey / np.sqrt(norm)

    V_out = S @ V_inc  # outgoing fields
    V_r = V_out[
        : 2 * (2 * n_mod_y + 1) * (2 * n_mod_x + 1)
    ]  # Just the reflected fields
    V_t = V_out[
        2 * (2 * n_mod_y + 1) * (2 * n_mod_x + 1) :
    ]  # Just the transmitted fields
    reflechi = compute2D.rt_efficiency(
        perm_top, k0, kx, ky, struct.periodx, struct.periody, ext, V_r
    )
    transm = compute2D.rt_efficiency(
        perm_top, k0, kx, ky, struct.periodx, struct.periody, ext, V_t
    )

    # We want the main order, but we could look for others
    r = reflechi[0]
    t = transm[0]

    kz_t = np.real(Vs[-1][n_mod])
    R = np.abs(r**2)
    T = np.abs(t) ** 2 * kz_t / (k0 * np.cos(theta)) * (perm_top / perm_bot)

    return r, t, R, T


def local_f_prime(a, b, eta, x):
    """
    compute f' on a given segmen
    """
    mask = (a <= np.array(x)) * (np.array(x) < b)
    f_prime = 1 - eta * np.cos(2 * np.pi * (x - a) / (b - a))
    f_prime = f_prime * mask
    return f_prime


def f_prime(interf, eta, x):
    """
    compute f' (the stretching function) along a given direction
    """
    f = np.zeros_like(x)
    for i in range(len(interf) - 1):
        a, b = interf[i], interf[i + 1]
        f += local_f_prime(a, b, eta, x)
    f[-1] = 1 - eta
    return f


def layer_field(D_minus, U_minus, D_plus, U_plus, V, P, ny, nx, h, kx, n_mod):
    """
    Docstring for field

    :param S_down: Description
    :param S_up: Description
    :param V: Description
    :param P: Description
    :param ny: Description
    :param h: Description
    :param kx: Description
    :param n_mod: Description
    """
    n_term = int(np.shape(D_minus)[0] / 2)

    n_mod_total = 2 * n_mod + 1
    exc = np.zeros(n_mod_total)  # excitation
    exc[n_mod] = 1

    """
        At this point, S_down is D-, S_up is U+
        So to get A-, I need to cascade U+ one last time
        And to get B+, I need to cascade D- one last time
    """
    layer_A = base.intermediaire(U_minus, D_minus, mode="A") @ exc
    layer_B = base.intermediaire(U_plus, D_plus, mode="B") @ exc

    # The field values, computed at each position in the layer
    M = np.zeros((ny, nx), dtype=complex)

    kxs = 2 * np.pi * np.arange(-n_mod, n_mod + 1)
    x = np.linspace(0, 1, nx)

    for k in range(ny):
        y = h / ny * k
        phase_up = np.diag(np.exp(1j * V * (h - y)))
        phase_down = np.diag(np.exp(1j * V * y))
        A_phase = phase_up @ layer_A
        B_phase = phase_down @ layer_B
        Fourier = P[:n_term] @ (A_phase + B_phase)  # Fourier decomposition of the field
        for i in range(len(kxs)):
            M[k, :] += Fourier[i] * np.exp(1.0j * x * kxs[i])
            
    M = M * np.exp(1j * kx * x)

    return M  # /np.abs(np.max(M))


def compute_field_1D(struct, wavelength, incidence, z_res, xres, n_mod, PV=None):
    """
    Hopefully, computing fields
    """

    interfaces = np.array(struct.interfaces) / struct.period
    wavelength_norm = wavelength / struct.period

    theta, pol = incidence  # In 1D, only one angle of incidence is necessary
    k0 = 2 * np.pi / wavelength_norm
    kx = k0 * np.sin(theta)

    if not PV is None:
        print("Ps and Vs were given")
        Ps, Vs = PV
    else:
        Ps, Vs = compute1D.compute_PV(
            struct, wavelength, interfaces, k0, kx, pol, n_mod
        )

    # Normalisation

    thickness = np.array(
        [wavelength_norm if t == 0 else t / struct.period for t in struct.thicknesses]
    )

    n_mod_total = 2 * n_mod + 1

    n_layers = len(struct.thicknesses)

    # Neutral S matrix
    S11 = np.zeros((n_mod_total, n_mod_total))
    S12 = np.eye(n_mod_total)
    S0 = np.block([[S11, S12], [S12, S11]])

    # Interface matrices
    I = []
    for k in range(n_layers - 1):  # nlayer - 1 interfaces in the structure
        I.append(base.interface(Ps[k], Ps[k + 1]))

    # Intermediate S matrices starting from the top
    U_plus = []
    U_minus = []
    # We use the neutral S matrix at the top, because the fields are already counted from the top
    U_plus.append(S0)
    U_minus.append(base.layer(Vs[0], thickness[0]))
    for k in range(n_layers - 1):
        # matlab ref:
        S_new = base.cascade(U_plus[k], base.c_up(I[k], Vs[k], thickness[k]))
        U_plus.append(S_new)
        U_minus.append(base.c_down(S_new, Vs[k + 1], thickness[k + 1]))

    # Intermediate S matrices starting from the bottom
    D_minus = []
    D_plus = []
    D_minus.append(S0)
    D_plus.append(base.layer(Vs[-1], thickness[-1]))

    for k in range(n_layers - 1, 0, -1):
        S_new = base.cascade(base.c_down(I[k - 1], Vs[k], thickness[k]), D_minus[0])
        D_minus.insert(0, S_new)
        D_plus.insert(0, base.c_up(S_new, Vs[k - 1], thickness[k - 1]))

    ny = np.floor(thickness * struct.period / z_res)
    nx = int(np.floor(struct.period / xres))
    print("sizes", ny, nx)

    M = layer_field(
        np.array(D_minus[0]),
        np.array(U_minus[0]),
        np.array(D_plus[0]),
        np.array(U_plus[0]),
        np.array(Vs[0]),
        np.array(Ps[0]),
        int(ny[0]),
        nx,
        thickness[0],
        kx,
        n_mod,
    )

    for j in np.arange(1, n_layers):
        M_new = layer_field(
            np.array(D_minus[j]),
            np.array(U_minus[j]),
            np.array(D_plus[j]),
            np.array(U_plus[j]),
            np.array(Vs[j]),
            np.array(Ps[j]),
            int(ny[j]),
            nx,
            thickness[j],
            kx,
            n_mod,
        )
        M = np.append(M, M_new, 0)
    M = np.array(M)
    xs = np.linspace(0, struct.period, nx)[::-1]
    zs = np.linspace(0, sum(struct.thicknesses), int(sum(ny)))[::-1]
    return xs, zs, M