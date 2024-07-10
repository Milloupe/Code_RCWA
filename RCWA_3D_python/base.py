import numpy as np
import scipy.linalg as lin

def cascade(T, U):
    '''Cascading of two scattering matrices T and U.
    Since T and U are scattering matrices, it is expected that they are square
    and have the same dimensions which are necessarily EVEN.
    '''
    n = int(T.shape[0]/2)
    T11 = T[0:n, 0:n]
    T12 = T[0:n, n:2*n]
    T21 = T[n:2*n, 0:n]
    T22 = T[n:2*n, n:2*n]

    U11 = U[0:n, 0:n]
    U12 = U[0:n, n:2*n]
    U21 = U[n:2*n, 0:n]
    U22 = U[n:2*n, n:2*n]

    J = np.linalg.inv(np.eye(n) - U11 @ T22)
    K = np.linalg.inv(np.eye(n) - T22 @ U11)
    
    S = np.block([[T11 + T12 @ J @ U11 @ T21, T12 @ J @ U12],
                  [U21 @ K @ T21, U22 + U21 @ K @ T22 @ U12]])
    return S


def c_bas(A, V, h):
    ''' Directly cascading any scattering matrix A (square and with even
    dimensions) with the scattering matrix of a layer of thickness h in which
    the wavevectors are given by V. Since the layer matrix is
    essentially empty, the cascading is much quicker if this is taken
    into account.
    '''
    n = int(A.shape[0]/2)
    D = np.diag(np.exp(1.0j*V*h))
    S = np.block([[A[0:n, 0:n], A[0:n, n:2*n] @ D],
                  [D @ A[n:2*n, 0:n], D @ A[n:2*n, n:2*n] @ D ]])
    return S

# Not used in this version!
# def c_haut(A, valp, h):
#     n = int(A[0].size/2)
#     D = np.diag(np.exp(1.0j*valp*h))
#     S11 = D @ A[0:n, 0:n] @ D
#     S12 = D @ A[0:n, n:2*n]
#     S21 = A[n:2*n, 0:n] @ D
#     S22 = A[n:2*n, n:2*n]
#     S = np.block([[S11, S12],
#                   [S21, S22]])
#     return S    

# Not used in this version!
# def intermediaire(T, U):
#     n = T.shape[0] // 2
#     H = np.linalg.inv( np.eye(n) - U[0:n, 0:n] @ T[n:2*n, n:2*n])
#     K = np.linalg.inv( np.eye(n) - T[n:2*n, n:2*n] @ U[0:n, 0:n])
#     a = K @ T[n:2*n, 0:n]
#     b = K @ T[n:2*n, n:2*n] @ U[0:n, n:2*n]
#     c = H @ U[0:n, 0:n] @ T[n:2*n, 0:n]
#     d = H @ U[0:n, n:2*n]
#     S = np.block([[a, b],
#                   [c, d]])
#     return S

# Not used in this version!
# def couche(valp, h):
#     n = len(valp)
#     AA = np.diag(np.exp(1.0j*valp*h))
#     C = np.block([[np.zeros((n, n)), AA], [AA, np.zeros((n, n))]])
#     return C


def interface(P, Q):
    """
        Computation of the scattering matrix of an interface, P and Q being the
        matrices given for each layer by homogene, reseau or creneau.
        P[:n,   i] is the x component of the mode i
        P[n:2:, i] is the y component of the mode i
        The S matrix computed is simply the solution to the continuity relations over the
        EM field at the interface
    """
    n = int(P.shape[1])
    A = np.block([[P[0:n, 0:n], -Q[0:n, 0:n]],
                  [P[n:2*n, 0:n], Q[n:2*n, 0:n]]])
    B = np.block([[-P[0:n, 0:n], Q[0:n, 0:n]],
                  [P[n:2*n, 0:n], Q[n:2*n, 0:n]]])
    S = np.linalg.inv(A) @ B
    return S



def eps11(s):
    """
        eps * g(y) / f(x)
    """
  
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, len(s.eps)), dtype=complex)
    for l in range(len(s.eps)):
        v = 0
        for j in range(len(s.eps[0])):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + 1 / s.eps[l, j] * tfx * (1 + 1.0j * s.pmlx[j]) # f / eps
        v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
        T[:, :, l] = np.linalg.inv(toep(v)) # eps / f

    M = np.zeros((m*n, m*n), dtype=complex)
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(len(s.eps)):
                tfy = tfd(s.oy[l], s.oy[l+1], s.ny[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l]) # eps / f * g
            M[j*n:(j+1)*n, k*n:(k+1)*n] = toep(v) # eps / f * g

    return M


def eps22(s):
    """
        eps * f(x) / g(y)
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, len(s.eps)), dtype=complex)
    for l in range(len(s.eps)):
        v = 0
        for j in range(len(s.eps[0])):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + s.eps[l, j] * tfx * (1 + 1.0j * s.pmlx[j]) # eps * f
        v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
        T[:, :, l] = toep(v) # eps * f

    M = np.zeros((m*n, m*n), dtype=complex)
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(len(s.eps)):
                tfy = tfd(s.oy[l], s.oy[l+1], s.ny[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + 1 / T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l])  # 1 / (eps * f) * g
            M[j*n:(j+1)*n, k*n:(k+1)*n] = np.linalg.inv(toep(v)) # eps * f / g

    return M


def eps33(s):
    """
        eps * f(x) * g(y)
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, len(s.eps)), dtype=complex)
    for l in range(len(s.eps)):
        v = 0
        for j in range(len(s.eps[0])):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
    
            v = v + s.eps[l, j] * tfx * (1 + 1.0j * s.pmlx[j]) # eps * f
    
        v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
        T[:, :, l] = toep(v) # eps * f
    M = np.zeros((m*n, m*n), dtype=complex)
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(len(s.eps)):
                tfy = tfd(s.oy[l], s.oy[l+1], s.ny[l], s.ny[l+1], s.eta, s.ny[-1], n)
        
                v = v + T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l]) # eps * f * g
            M[j*n:(j+1)*n, k*n:(k+1)*n] = toep(v) # eps * f * g
    
    return M


def mu11(s):
    """
        mu * g(y) / f(x)
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, len(s.eps)), dtype=complex)
    for l in range(len(s.eps)):
        v = 0
        for j in range(len(s.eps[0])):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + 1 / s.mu[l, j] * tfx * (1 + 1.0j * s.pmlx[j]) # f / mu
        v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
        T[:, :, l] = np.linalg.inv(toep(v)) # mu / f

    M = np.zeros((m*n, m*n), dtype=complex)
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(len(s.eps)):
                tfy = tfd(s.oy[l], s.oy[l+1], s.ny[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l]) # mu / f * g
            M[j*n:(j+1)*n, k*n:(k+1)*n] = toep(v) # mu / f * g

    return M


def mu22(s):
    """
        mu * f(x) / g(y)
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, len(s.eps)), dtype=complex)
    for l in range(len(s.eps)):
        v = 0
        for j in range(len(s.eps[0])):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
    
            v = v + s.mu[l, j] * tfx * (1 + 1.0j * s.pmlx[j]) # mu * f

        v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
        T[:, :, l] = toep(v) # mu * f

    M = np.zeros((m*n, m*n), dtype=complex)
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(len(s.eps)):
                tfy = tfd(s.oy[l], s.oy[l+1], s.ny[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + 1 / T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l]) # 1  / (mu * f) * g
            M[j*n:(j+1)*n, k*n:(k+1)*n] = np.linalg.inv(toep(v)) # mu * f / g

    return M


def mu33(s):
    """
        mu * f(x) * g(y)
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    T = np.zeros((m, m, len(s.eps)), dtype=complex)
    for l in range(len(s.eps)):
        v = 0
        for j in range(len(s.eps[0])):
            tfx = tfd(s.ox[j], s.ox[j+1], s.nx[j], s.nx[j+1], s.eta, s.nx[-1], m)
            v = v + s.mu[l, j] * tfx * (1 + 1.0j * s.pmlx[j]) # mu * f
        v = v * (np.abs(v) > 1e-12 * np.max(np.abs(v)))
        T[:, :, l] = toep(v) # mu * f

    M = np.zeros((m*n, m*n), dtype=complex)
    for j in range(m):
        for k in range(m):
            v = 0
            for l in range(len(s.eps)):
                tfy = tfd(s.oy[l], s.oy[l+1], s.ny[l], s.ny[l+1], s.eta, s.ny[-1], n)
                v = v + T[j, k, l] * tfy * (1 + 1.0j * s.pmly[l]) # mu * f * g
            M[j*n:(j+1)*n, k*n:(k+1)*n] = toep(v) # mu * f * g

    return M


def g(s, y):
    """
        Stretching function along y
    """
    j = np.argmax((s.ny-y)>0) - 1

    new_diff = s.ny[j+1]-s.ny[j]
    new_k = 2*np.pi/new_diff
    old_diff = s.oy[j+1]-s.oy[j]

    val = s.oy[j] + old_diff/new_diff * (y - s.ny[j] - s.eta * np.sin(new_k*(y-s.ny[j])) / new_k)
    return val


def f(s, x):
    """
        Stretching function along x
    """
    j = np.argmax((s.nx-x)>0) - 1

    new_diff = s.nx[j+1]-s.nx[j]
    new_k = 2*np.pi/new_diff
    old_diff = s.ox[j+1]-s.ox[j]

    val = s.ox[j] + old_diff/new_diff * (x - s.nx[j] - s.eta * np.sin(new_k*(x-s.nx[j])) / new_k)
    return val
  

def reseau(s):
    """
        Computes the modes and eigenvalues in a structured layer
    """

    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1

    v_a = []
    for j in range(-s.Mm, s.Mm+1):
        a = s.kx + 2*np.pi*j / s.nx[-1]
        v_a.extend([a] * n)
    alpha = 1.0j*np.diag(v_a)

    v_b = s.ky + 2*np.pi * np.arange(-s.Nm, s.Nm+1) / s.ny[-1]
    v_b = np.tile(v_b, (m))
    beta = 1.0j*np.diag(v_b)

    inv_e33 = np.linalg.inv(eps33(s))
    inv_e33 = inv_e33 / (1.0j*s.k0)
    alpha_eps_beta = alpha @ inv_e33 @ beta 
    alpha_eps_alpha = alpha @ inv_e33 @ alpha
    beta_eps_beta = beta @ inv_e33 @ beta
    beta_eps_alpha = beta @ inv_e33 @ alpha
    Leh = np.block([[alpha_eps_beta, 1.0j*s.k0*mu22(s) - alpha_eps_alpha],
                    [-1.0j*s.k0*mu11(s) + beta_eps_beta, -beta_eps_alpha]])
    

    inv_mu33 = np.linalg.inv(mu33(s)) / (1.0j*s.k0)
    alpha_mu_beta = alpha @ inv_mu33 @ beta
    alpha_mu_alpha = alpha @ inv_mu33 @ alpha
    beta_mu_beta = beta @ inv_mu33 @ beta
    beta_mu_alpha = beta @ inv_mu33 @ alpha
    Lhe = np.block([[-alpha_mu_beta, -1.0j*s.k0*eps22(s) + alpha_mu_alpha],
                    [1.0j*s.k0*eps11(s)-beta_mu_beta, beta_mu_alpha]])

    L = Leh @ Lhe
    L = L * (np.abs(L)>1e-18)
    [V, inv_L] = np.linalg.eig(L)
    # for i_vp in range(inv_L.shape[1]):
    #     inv_L[:, i_vp] = inv_L[:, i_vp]/np.exp(1.0j*np.angle(inv_L[0, i_vp]))
    
    V = np.sqrt(-V)
    
    neg_val = np.imag(V) < 0
    V = V * (1 - 2*neg_val)
    # keep_im = np.abs(np.imag(V)) > 1e-15*np.abs(np.real(V))
    # V = np.real(V) + 1.0j*np.imag(V)*keep_im
    P = np.block([[inv_L],
                  [Lhe @ inv_L @ np.diag(1 / V)]])
    

    return (P, V)


def genere(ox, nx, eta, n):
    """
        Computes the fourier transform of coordinates, with stretching,
        for all interface coordinates
    """
    fp = []
    for i in range(len(ox)-1):
        fp.append(tfd(ox[i], ox[i+1], nx[i], nx[i+1], eta, nx[-1], n).T)
    return np.array(fp).T # TODO: check whether transposing is necessary


def homogene(s, ext=0):
    """
        Computing modes and eignevalues in a homogeneous layer
        Takes into account the possibility that it is the first or last layer
        (ext), in which case we match with propagative modes
    """
    n = 2 * s.Nm + 1
    m = 2 * s.Mm + 1
    global nb_mod # Why did I have to for this ??
    
    v_a = []
    for j in range(-s.Mm, s.Mm+1):
        a = s.kx + 2*np.pi*j/s.nx[-1]
        v_a.extend([a] * n)
    alpha = 1.0j*np.diag(v_a)

    v_b = s.ky + 2*np.pi*np.arange(-s.Nm, s.Nm+1)/s.ny[-1]
    v_b = np.tile(v_b, (m))
    beta = 1.0j*np.diag(v_b)

    i_eps33 = np.linalg.inv(eps33(s))
    i_mu33 = np.linalg.inv(mu33(s))
    eps1 = eps11(s)
    eps2 = eps22(s)
    mu1 = mu11(s)
    mu2 = mu22(s)
    L = -s.k0**2 * mu2 @ eps1 - alpha @ i_eps33 @ alpha @ eps1 - mu2 @ beta @ i_mu33 @ beta
    [B, A] = np.linalg.eig(L)

    L = -s.k0**2 * mu1 @ eps2 - beta @ i_eps33 @ beta @ eps2 - mu1 @ alpha @ i_mu33 @ alpha

    [D, C] = np.linalg.eig(L)

    E = np.block([[A, np.zeros((n*m, n*m))],
                  [np.zeros((n*m, n*m)), C]])

    inv_mu33 = i_mu33 / (1.0j*s.k0)
    alpha_mu_beta = alpha @ inv_mu33 @ beta
    alpha_mu_alpha = alpha @ inv_mu33 @ alpha
    beta_mu_beta = beta @ inv_mu33 @ beta
    beta_mu_alpha = beta @ inv_mu33 @ alpha
    Lhe = np.block([[-alpha_mu_beta, -1.0j*s.k0*eps22(s) + alpha_mu_alpha],
                    [1.0j*s.k0*eps11(s)-beta_mu_beta, beta_mu_alpha]])

    V = np.block([B, D])

    if (ext):
        # This layer is a substrate or superstrate, ext is provided
        # -> we are interested in the Rayleigh decomposition of the modes

        V = np.sqrt(-V)
        neg_val = np.angle(V) < -np.pi/2 + 1e-5
        V = V * (1 - 2*neg_val)



        # Finding real eigen values and their positions in V
        nb_real = np.sum(abs(np.angle(V))<1e-4)
        ana_kz = np.zeros((4, nb_real), dtype=complex)
        j=0
        for i in range(len(V)):
            if (abs(np.angle(V[i]))<1e-4):
                ana_kz[3, j] = i
                j += 1
        p = j

        # Compute the analytical eigen values (Rayleigh decomposition)

        dx = s.nx[-1]
        kx = 2*np.pi/dx
        dy = s.ny[-1]
        ky = 2*np.pi/dy

        k = np.sqrt(s.eps[0, 0]*s.mu[0, 0])*s.k0 # DEBUGG CHANGED [1, 1] TO [0, 0]

        min_ord_x = int((k + s.kx)/kx)
        max_ord_x = int((k - s.kx)/kx)
        min_ord_y = int((k + s.ky)/ky)
        max_ord_y = int((k - s.ky)/ky)

        # TODO: check is correct, changed max ord from dx to 1/kx
        nb_ana = 0
        for nx in range(-min_ord_x, max_ord_x+1):
            for ny in range(-min_ord_y, max_ord_y+1):
                gamma = np.sqrt(0j+k**2-(s.kx + nx*kx)**2-(s.ky + ny*ky)**2)
                # Computes the (ny, nx) diffracted order
                if (np.abs(np.angle(gamma))<1e-4):
                    # Keeping only propagative modes
                    ana_kz[0, nb_ana+1] = gamma
                    ana_kz[1, nb_ana+1] = nx
                    ana_kz[2, nb_ana+1] = ny
                    nb_ana = nb_ana+1
                    if ((ny == 0) and (nx == 0)):
                        ana_kz[:3, 0] = ana_kz[:3, nb_ana]
                        nb_ana = nb_ana-1
        # ana_kz = np.array(ana_kz).T
        nb_ana += 1


        # if (np.shape(position)[1] == 2*np.shape(ana_kz)[1]):
        #     ana_kz = np.block([ana_kz, ana_kz])
        #     ana_kz[4, :] = position
        # else:
        if (2*nb_ana != p):
            print('Missing modes! (homogene) expected: ', 2*nb_ana, ' found: ', p)
        
        # DEBUGG fortran code "on réplique"
        ana_kz[:3, nb_ana:2*nb_ana] = ana_kz[:3, :nb_ana]

        for i_mod in range(2*nb_ana):
            # If the mode is propagative, it is in ana_kz
            # and we replace it in V (more precise?)
            V[int(ana_kz[3, i_mod])] = ana_kz[0, i_mod]


        ana_kz[0, 0] = nb_ana # storing nb_mod of mdoes

        # Replacing modes

        n = 2 * s.Nm + 1
        m = 2 * s.Mm + 1

        k=n*m        
        x=0.
        for k in range(np.shape(s.eps)[1]):
            x=x+tfd(s.ox[k], s.ox[k+1], s.nx[k], s.nx[k+1], s.eta, s.nx[-1], m)
        tmp = toep(x)
        

        unite = np.eye(n)
        alpha = np.zeros((n*m, n*m), dtype=complex)
        for j in range(m):
            for k in range(m):
                alpha[j*n:(j+1)*n, k*n:(k+1)*n] = tmp[j, k]*unite
        y=0.
        for k in range(np.shape(s.eps)[0]):
            y=y+tfd(s.oy[k], s.oy[k+1], s.ny[k], s.ny[k+1], s.eta, s.ny[-1], n)
        tmp = toep(y)

        beta = np.zeros((n*m, n*m), dtype=complex)
        for j in range(m):
                beta[j*n:(j+1)*n, j*n:(j+1)*n] = tmp
        for j in range(int(ana_kz[0, 0])):

            nb_mod = 2048
            pos_x = np.arange(nb_mod)/nb_mod*dx
            x = np.zeros(nb_mod, dtype=complex)
            for k in range(nb_mod):
                x[k] = np.exp(1.0j*(s.kx + ana_kz[1, j]*kx)*f(s, pos_x[k])-1.0j*s.kx*pos_x[k])
            x = np.fft.fft(x)/nb_mod
            x = np.block([x[nb_mod-s.Mm:nb_mod], x[:s.Mm+1]])


            pos_y = np.arange(nb_mod)/nb_mod*dy
            y = np.zeros(nb_mod, dtype=complex)
            for k in range(nb_mod):
                y[k] = np.exp(1.0j*(s.ky + ana_kz[2, j]*ky)*g(s, pos_y[k])-1.0j*s.ky*pos_y[k])
            y = np.fft.fft(y)/nb_mod
            y = np.block([y[nb_mod-s.Nm:nb_mod], y[:s.Nm+1]])


            vtmp = np.zeros(m*n, dtype=complex)
            for k in range(2 * s.Mm + 1):
                l = k*n
                vtmp[l:l+n] = x[k]*y
            # vtmp = np.array([vtmp]).T

            E[:n*m, int(ana_kz[3, j])] = alpha @ vtmp
            E[n*m:, int(ana_kz[3, j + np.shape(ana_kz)[1]//2])] = beta @ vtmp
        P = np.block([[E],
                    [Lhe @ E @ np.diag(1 / V)]])
        return (P, V), ana_kz

    else:
        # Not in a substrate/superstrate, we simply keep the modes
        V = np.sqrt(-V)
        neg_val = np.imag(V) < 0
        V = V * (1 - 2*neg_val)
        ana_kz = []
    P = np.block([[E],
                  [Lhe @ E @ np.diag(1 / V)]])
    return (P, V)


def efficace(a, ext, E):
    # Computing the total reflected or transmitted energy (depending on the provided vector E)
    nb_mod = int(np.real(ext[0, 0]))
    res = np.copy(ext[:4, :nb_mod])
    k2 = a.eps[0, 0] * a.mu[0, 0] * a.k0**2

    for i in range(nb_mod):
       kxn = a.kx + 2*np.pi*ext[1, i] / a.nx[-1]

       kyn = a.ky + 2*np.pi*ext[2, i] / a.ny[-1]

       ind1 = int(np.real(ext[3, i]))
       ind2 = int(np.real(ext[3, i+nb_mod]))

       A = (k2-kyn**2) * np.abs(E[ind1])**2
       B = (k2-kxn**2) * np.abs(E[ind2])**2
       C = 2*kxn*kyn * np.real(E[ind1]*np.conj(E[ind2]))
       denom = a.mu[0, 0]*ext[0, i+nb_mod]

       res[3, i] = (A+B+C) / denom
    return res



def tfd(old_a, old_b, new_a, new_b, eta, d, N):
    """
        Computing fourier transform of coordinates with stretching
    """
    pi = np.pi
    fft = np.zeros(2*N+1, dtype=complex)
    old_ba = old_b - old_a
    new_ba = new_b - new_a
    k_na = 2.0j*pi*new_a/d
    k_nb = 2.0j*pi*new_b/d
    prefac = -1 / (2.0j*pi) * old_ba/new_ba
    for i_mod in range(-N, N+1):
        sinc_prefac = old_ba * i_mod / d * np.sinc(i_mod/d * new_ba) * np.exp(-1.0j*np.pi*i_mod * (new_b+new_a) / d)
        exp_kb = np.exp(-k_nb*i_mod)
        exp_ka = np.exp(-k_na*i_mod)

        n_diff = i_mod*new_ba
        if (i_mod == 0):
            fft[N] = old_ba / d
        elif (d-n_diff == 0):
            # fft[i_mod + N] = prefac * (1/i_mod - eta/2 * new_ba/(d+n_diff)) * (exp_kb-exp_ka) - eta/2 * old_ba*np.exp(-2.0j*pi*new_a/new_ba)/d
            fft[i_mod + N] = sinc_prefac * (1/i_mod - eta/2 * new_ba/(d+n_diff)) - eta/2 * old_ba*np.exp(-2.0j*pi*new_a/new_ba)/d
        elif (d+n_diff == 0):
            # fft[i_mod + N] = prefac * (1/i_mod + eta/2 * new_ba/(d-n_diff)) * (exp_kb-exp_ka) - eta/2 * old_ba*np.exp(2.0j*pi*new_a/new_ba)/d
            fft[i_mod + N] = sinc_prefac * (1/i_mod + eta/2 * new_ba/(d-n_diff)) - eta/2 * old_ba*np.exp(2.0j*pi*new_a/new_ba)/d
        else:
            # fft[i_mod + N] = prefac * (1/i_mod + eta/2 * (new_ba/(d-n_diff)-new_ba/(d+n_diff))) * (exp_kb-exp_ka)
            fft[i_mod + N] = sinc_prefac * (1/i_mod + eta/2 * (new_ba/(d-n_diff)-new_ba/(d+n_diff)))
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
    n = (len(v)-1)//2
    a = v[n:0:-1]
    b = v[n:2*n]
    T = lin.toeplitz(b, a)
    return T