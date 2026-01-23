import numpy as np
import RCWA_project.compute1D as compute1D
import RCWA_project.base as base

""" TODO:
    - create r, t, R, T functions for 1D, 1D angle and 2D
    - create Field functions for 1D, 1D angle and 2D
    - create wrappers that automatically detect which should be called
    (criteria: structure in y has only one zone -> 1D, no phi angle -> pure 1D)
    -> eventually, the 'structure' class will let me check the validity of the structure!
"""

def compute_PV(struct, wavelength, interfaces, k0, kx, pol, Mm):
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

    for ilayer in range(nb_layer):
        layer = struct.layers[ilayer]
        homo = struct.homo_layer[ilayer]
        if homo:
            epsilon = layer[0].get_permittivity(wavelength)
            eig_vec, eig_val = compute1D.homogeneous(epsilon, k0, kx, pol, Mm)
        else:
            epsilons = [mat.get_permittivity(wavelength) for mat in layer]
            eig_vec, eig_val = compute1D.structured(epsilons, interfaces, k0, kx, pol, Mm) 

        Ps.append(eig_vec)
        Vs.append(eig_val)
    return Ps, Vs


def coefficient_1D(struct, wavelength, incidence, n_mod):
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
    k0 = k0
    kx = kx

    # General structure parameters
    interfaces = np.array(struct.interfaces) / struct.period
    nb_layer = len(struct.thicknesses)
    # pmls = struct.pmls # So far, no pmls in 1D because no stretching

    Ps, Vs = compute_PV(struct, wavelength, interfaces, k0, kx, pol, n_mod)

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


def coefficient_1D_angle(structure, wavelength, incidence):
    """
    This function computes the reflection and transmission coefficients
    of a 1D structure when the incidence plane is not the same as the structured plane.

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


def coefficient_2D(structure, wavelength, incidence):
    """
    This function computes the reflection and transmission coefficients
    of a 2D structure.

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


def layer(V, h):
    """
    Computation of the scattering matrix of a layer (just the propagation)
    """
    n = len(V)
    AA = np.diag(np.exp(1j * V * h))
    C = np.block([[np.zeros((n, n)), AA], [AA, np.zeros((n, n))]])
    return C


def intermediaire(U, D, mode="A"):
    """
    Cascading of two scattering matrices, U and D, but specifically when computing
    intermediate coefficients, and therefore some elements are 0.
    HDR 1.44
    """
    n = U.shape[0] // 2
    U11 = U[n : 2 * n, n : 2 * n]
    D00 = D[0:n, 0:n]
    U10 = U[n : 2 * n, 0:n]

    if mode == "A":
        # Downwards coeff=
        S = np.linalg.inv(np.eye(n) - D00 @ U11) @ D00 @ U10
    
    elif mode == "B":
        # Upwards coeff
        S = np.linalg.inv(np.eye(n) - U11 @ D00) @ U10

    return S


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
    
    n_mod_total = 2*n_mod + 1
    exc = np.zeros(n_mod_total)  # excitation
    exc[n_mod] = 1

    """
        At this point, S_down is D-, S_up is U+
        So to get A-, I need to cascade U+ one last time
        And to get B+, I need to cascade D- one last time
    """
    layer_A = intermediaire(U_minus, D_minus, mode="A") @ exc
    # X = intermediaire(S_up, S_down) @ exc
    # layer_A = X[0:n_term]
    layer_B = intermediaire(U_plus, D_plus, mode="B") @ exc
    # layer_B = X[n_term:2*n_term]
    # print("Debugg layer_field, A ", np.angle(layer_A))
    # print("Debug layer_field, layer", S_down, S_up)
    # print("Debugg layer_field, B ", layer_B)

    # The field values, computed at each position in the layer
    M = np.zeros((ny,nx), dtype = complex)

    # print("Debugg layer_field, M shape", np.shape(M), "P shape", np.shape(P))
    # print("Debugg layer_field, S shape", np.shape(S_layer), np.shape(S_up), np.shape(S_down))


    kxs = 2 *  np.pi * np.arange(-n_mod, n_mod + 1)
    # print("Debugg layer_field, V", V[n_mod], ", kxs", kxs, kxs[n_mod])
    x = np.linspace(0, 1, nx)
    # print("DEBUGG layer_field V", V)
    
    for k in range(ny):
        y = h / ny * k
        # Fourier = np.matmul(P,np.matmul(np.diag(np.exp(1j*V*y)),layer_A) + np.matmul(np.diag(np.exp(1j*V*(h-y))),layer_B))
        phase_up = np.diag(np.exp(1j * V * (h-y)))
        phase_down = np.diag(np.exp(1j * V * y))
        A_phase = phase_up @ layer_A
        B_phase = phase_down @ layer_B
        Fourier = P[:n_term] @ (A_phase + B_phase) # Fourier decomposition of the field
        for i in range(len(kxs)):
            M[k,:] += Fourier[i] * np.exp(1.0j * x * kxs[i])
        # print("Debug layer_field phasedown", np.diag(phase_down))#, M[k,:])

        # # temp = np.fft.ifftshift(Fourier[0:len(Fourier)-1])
        # print(f"Debugg layer_field, x shape {np.shape(x)}, kxs shape {np.shape(kxs)}, Fourier shape {np.shape(Fourier)}")
        # temp = Fourier * np.exp(1.0j * x.T @ kxs)
        # print(f"Debugg layer_field, axis0 {np.shape(np.sum(temp, axis=0))}, axis1 {np.shape(np.sum(temp, axis=1))}")
        # M[k,:] = np.sum(Fourier * np.exp(1.0j * x * kxs))

    # M = np.conj(np.fft.ifft(np.conj(M).T, axis = 0)).T * nx
    # x, y = np.meshgrid(np.linspace(0,1,nx-1), np.linspace(0,1,ny))
    # # We multiply by the phase along x direction (no influence if we plot only the field module)
    M = M * np.exp(1j * kx * x)

    return(M)#/np.abs(np.max(M))


def compute_field_1D(struct, wavelength, incidence, z_res, xres, n_mod, PV=None):
    """
    Hopefully, computing fields
    """

    interfaces = np.array(struct.interfaces) / struct.period
    wavelength_norm = wavelength / struct.period

    theta, pol = incidence  # In 1D, only one angle of incidence is necessary
    k0 = 2 * np.pi / wavelength_norm
    kx = k0 * np.sin(theta)
    # print(f"Debugg compute_field, k0 {k0}, , kx {kx}, kx/k0 {kx/k0}, period {struct.period}, wav {wavelength}")

    if not PV is None:
        print("Ps and Vs were given")
        Ps, Vs = PV
    else:
        Ps, Vs = compute_PV(struct, wavelength, interfaces, k0, kx, pol, n_mod)

    # Normalisation

    thickness = np.array([wavelength_norm if t == 0 else t/struct.period for t in struct.thicknesses])

    n_mod_total = (2 * n_mod + 1)  

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
    U_plus.append(S0) # We use the neutral S matrix at the top, because the fields are already counted from the top
    U_minus.append(layer(Vs[0], thickness[0]))
    for k in range(n_layers - 1):
        # matlab ref:
        # S{j+1} =cascade(S{j},c_haut(I{j},V{j},thickness(j)));
        S_new = base.cascade(U_plus[k], base.c_up(I[k], Vs[k], thickness[k]))
        U_plus.append(S_new)
        U_minus.append(base.c_down(S_new, Vs[k+1], thickness[k+1]))


    # Intermediate S matrices starting from the bottom
    D_minus = []
    D_plus = []
    D_minus.append(S0)
    D_plus.append(layer(Vs[-1], thickness[-1]))

    for k in range(n_layers - 1, 0, -1):
        # matlab ref : 
        # Q{j+1}=cascade(c_bas(I{n_couches-j},A{n_couches-j+1,2},thickness(n_couches-j+1)),Q{j});
        # n_lay_up = n_layers - (k + 1)
        # print("Debug compute_field_1D, S down", len(Vs), k)
        
        S_new = base.cascade(base.c_down(I[k-1], Vs[k], thickness[k]), D_minus[0])
        D_minus.insert(0, S_new)
        D_plus.insert(0, base.c_up(S_new, Vs[k-1], thickness[k-1]))

    # S_down.insert(0, base.c_up(S_down[0], Vs[0], thickness[0]))

    # print("Debugg compute_field_1D, shapes", len(U_plus), len(D_minus))

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
        n_mod
    )
    # print("Debugg compute_field_1D, ilayer 0, ,M", np.shape(M))

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
            n_mod
        )
        # print("Debugg compute_field_1D, ilayer", j, ", Mnew", np.shape(M_new))
        M = np.append(M, M_new, 0)
    M = np.array(M)
    # print("Debugg compute_field_1D, M phase", 
        #   M[0, 0],
        #   (M[int(ny[0]), 0]),
        #   (M[int(ny[0])-1, 0]))

    xs = np.linspace(0, struct.period, nx)[::-1]
    zs = np.linspace(0, sum(struct.thicknesses), int(sum(ny)))[::-1]
    return xs, zs, M

# def E_field(S_down, S_up, V, P, excitation, ny, h, alpha_0):
#     """
#     Hopefully, helps compute the fields
#     TODO: move this to 3D, or 2D slice (ouch)
#     """
#     n = int(np.shape(S_down)[0] / 2)
#     nmod = int((n - 1) / 2)
#     nx = n
#     exc = excitation.reshape(excitation.size, 1)

#     Vec = P[:n, :n]  # E field depends on the first half of the components

#     A1 = intermediaire(S_down, cascade(layer(V, h), S_up))
#     X = A1 @ exc
#     print("E_field", n, np.shape(X), np.shape(Vec))

#     layer_As = X[0:n]  # intensities of down-going fields

#     A2 = intermediaire(cascade(S_down, layer(V, h)), S_up)
#     X = A2 @ exc
#     layer_Bs = X[n : 2 * n]  # intensities of up-going fields

#     Ex = np.zeros((ny, nmod), dtype=complex)
#     Ey = np.zeros((ny, nmod), dtype=complex)
#     for k in range(ny):
#         y = h / ny * (k + 1)
#         phase_up = np.diag(np.exp(1j * V * y))
#         phase_down = np.diag(np.exp(1j * V * (h - y)))

#         print("test1", np.shape((phase_up @ layer_As + phase_down @ layer_Bs)))
#         print("test2", np.shape(Vec))
#         Fourier = Vec @ (phase_up @ layer_As + phase_down @ layer_Bs)
#         print("test3", np.shape(Fourier), Fourier)
#         MM = np.fft.ifftshift(Fourier[0 : len(Fourier) - 1])
#         print("test4", np.shape(MM), MM)
#         MM = MM.reshape(len(MM))

#     print(np.shape(np.fft.ifft(MM, axis=0)))
#     Ex = np.conj(np.fft.ifft(np.conj(Ex).T, axis=0)).T * n
#     x, y = np.meshgrid(np.linspace(0, 1, nmod), np.linspace(0, 1, ny))
#     Ex = Ex * np.exp(1j * alpha_0 * x)

#     Ey = np.conj(np.fft.ifft(np.conj(Ey).T, axis=0)).T * n
#     x, y = np.meshgrid(np.linspace(0, 1, nmod), np.linspace(0, 1, ny))
#     Ey = Ey * np.exp(1j * alpha_0 * x)

#     return Ex, Ey


# def Field_2D(
#     Ps, Vs, thickness, interf, eta, wavelength, angle, ext, z_res, x_res, n_mod, period
# ):
#     """
#     Hopefully, computing fields
#     QO:
#         - in 2D, we normalise wrt the period. Can we not do this in 3D?
#             Fix idea: if we only ever compute 2D maps (slices), then the period makes sens again
#     """

#     # Normalisation
#     wavelength_norm = wavelength / period

#     thickness = np.array([t / period for t in thickness])

#     k0 = 2 * np.pi / wavelength_norm
#     alpha_0 = k0 * np.sin(angle)

#     n_mod_total = 2 * (2 * n_mod + 1)  # Ex and Ey modes
#     # TODO: this is wrong ! But I don't know how to fix the rest (yet)

#     n_layers = thickness.size

#     # matrice neutre pour l'opération de cascadage
#     S11 = np.zeros((n_mod_total, n_mod_total))
#     S12 = np.eye(n_mod_total)
#     S0 = np.block([[S11, S12], [S12, S11]])
#     S_down = []
#     S_down.append(S0)

#     # matrices d'interface
#     B = []
#     for k in range(n_layers - 1):  # car nc - 1 interfaces dans la structure
#         c = interface(Ps[k], Ps[k + 1])
#         B.append(c)

#     # Matrices montantes
#     for k in range(n_layers - 1):
#         # a = np.array(S_down[k])
#         b = c_up(np.array(B[k]), np.array(Vs[k]), thickness[k])
#         S_new = cascade(S_down[k], b)
#         S_down.append(S_new)

#     a = np.array(S_down[n_layers - 1])
#     b = np.array(Vs[n_layers - 1])
#     c = c_down(a, b, thickness[n_layers - 1])
#     S_down.append(c.tolist())

#     # Matrices descendantes
#     S_up = []
#     S_up.append(S0)

#     for k in range(n_layers - 1):
#         a = np.array(B[n_layers - k - 2])
#         b = np.array(Vs[n_layers - (k + 1)])
#         c = thickness[n_layers - (k + 1)]
#         d = np.array(S_up[k])
#         Q_new = cascade(c_down(a, b, c), d)
#         S_up.append(Q_new)

#     # a = np.array(S_up[k])
#     # b = np.array(Vs[0])
#     c = c_up(S_up[-1], Vs[0], thickness[0])
#     S_up.append(c)

#     exc = np.zeros(2 * n_mod_total)  # excitation
#     # Eclairage par au dessus, onde plane
#     exc[ext] = 1
#     # eclairage par en dessous, onde plane
#     # exc[n_mod_total + n_mod] = 1
#     # eclairage par en dessous, guide d'onde (le mode avec la plus grande partie réelle)
#     # position = np.argmax(np.real(Vdown))

#     ny = np.floor(thickness * period / z_res)
#     print("sizes", ny)

#     M = E_field(
#         np.array(S_down[0]),
#         np.array(S_up[n_layers - 0 - 1]),
#         np.array(Vs[0]),
#         np.array(Ps[0]),
#         exc,
#         int(ny[0]),
#         thickness[0],
#         alpha_0,
#     )

#     for j in np.arange(1, n_layers):
#         M_new = E_field(
#             np.array(S_down[j]),
#             np.array(S_up[n_layers - j - 1]),
#             np.array(Vs[j]),
#             np.array(Ps[j]),
#             exc,
#             int(ny[j]),
#             thickness[j],
#             alpha_0,
#         )
#         M = np.append(M, M_new, 0)

#     nx = int(np.shape(S_down[0])[0] / 2)
#     xs_fft = np.linspace(0, 1, nx - 1) * period
#     xs_real = np.arange(0, period, x_res)
#     # print(np.shape(M), np.shape(xs_fft), nx, np.shape(S_down), xs_real)

#     z_len = len(M)
#     M_real = np.zeros((z_len, len(xs_real)), dtype=complex)
#     mem = np.array(M)
#     fprime = f_prime(interf, eta, xs_fft)
#     for i in range(z_len):
#         # print("pop", fprime)
#         M[i] = M[i] / fprime
#         M_real[i] = np.interp(xs_real, xs_fft, M[i], period=period)

#     # Mfield = np.abs(M) ** 2
#     return M, mem
