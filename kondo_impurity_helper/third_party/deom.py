import numpy as np
import sympy as sp
import os
from scipy.linalg import expm as matrix_exp
from .aux import fastread, PSD, fBose, direct_product_2d
import subprocess
from cvxopt import solvers, matrix
from numpy import linalg as LA
import importlib.util
import sys


def gen_twsg(expn):
    twsg = []
    syl_nind = 0
    expn_bk = expn.copy()
    syl_nmod = 0
    for i in expn:
        twsg_a = np.where(np.abs(expn_bk - i) < 1e-10)
        if (len(twsg_a[0]) > 0):
            twsg.append(twsg_a[0])
            syl_nind += 1
        syl_nmod = max(len(twsg_a[0]), syl_nmod)
        expn_bk[twsg_a] = -100
    for i in range(len(twsg)):
        if (len(twsg[i]) < syl_nmod):
            twsg[i] = np.append(twsg[i], np.zeros(
                syl_nmod - len(twsg[i]), dtype=int))
    twsg = np.array(twsg)
    return twsg, syl_nind, syl_nmod


def numpy_to_cvxopt_matrix(A):
    if A is None:
        return A
    if isinstance(A, np.ndarray):
        if A.ndim == 1:
            return matrix(A, (A.shape[0], 1), 'd')
        else:
            return matrix(A, A.shape, 'd')
    else:
        return A


def fit_t(t, res, expn, etal):
    for i in range(len(etal)):
        res += etal[i] * np.exp(-expn[i] * t)
    return res


def fit_J(w, res, expn, etal, sigma=-1):
    for i in range(len(etal)):
        res += etal[i] / (expn[i] + sigma * 1.j * w)


def decompose_spe_real(spe, w_sp, sp_para_dict, para_dict, condition_dict):
    numer, denom = sp.cancel(sp.factor(
        spe.subs(condition_dict))).as_numer_denom()
    numer_get_para = (sp.factor(numer)).subs(sp_para_dict)
    denom_get_para = (sp.factor(denom)).subs(sp_para_dict)
    poles = sp.nroots(denom_get_para)
    float(sp.re(poles[0]))

    expn, etal, etar, etaa = [], [], [], []
    poles_allplane = np.array([])
    for i in poles:
        i = complex(i)
        if i.imag < 0:
            expn.append(i * 1.J)
        poles_allplane = np.append(poles_allplane, i)

    expn = np.array(expn)
    expn_imag_sort = np.argsort(np.abs(np.imag(expn)))[::-1]
    expn_imag = np.sort(np.abs(np.imag(expn)))[::-1]
    expn_val_cc = expn[expn_imag_sort[expn_imag != 0]]
    expn_val_n_cc = expn[expn_imag_sort[expn_imag == 0]]

    for ii in range(0, len(expn_val_cc), 2):
        etal.append(
            complex(
                sp.N((-1.j * numer_get_para /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii]}))))

        etal.append(
            complex(
                sp.N((-1.j * numer_get_para /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii + 1]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii + 1]}))))

        etar.append(np.conj(etal[-1]))
        etar.append(np.conj(etal[-2]))
        etaa.append(np.sqrt(np.abs(etal[-2]) * np.abs(etar[-2])))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    for ii in range(len(expn_val_n_cc)):
        etal.append(
            complex(
                sp.N((-1.j * numer_get_para /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane -
                          -1.J * expn_val_n_cc[ii]) > 1e-14])).subs(
                              {w_sp: -1.j * expn_val_n_cc[ii]}))))
        etar.append(np.conj(etal[-1]))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    return np.array(etal), np.array(
        etar), np.array(etaa), np.array(expn)


def decompose_spe_imag(spe, w_sp, sp_para_dict, para_dict, condition_dict):
    numer, denom = sp.cancel(sp.factor(sp.cancel(
        spe.subs(condition_dict)))).as_numer_denom()
    numer_get_para = (sp.factor(numer)).subs(sp_para_dict)
    denom_get_para = (sp.factor(denom)).subs(sp_para_dict)

    poles = sp.nroots(denom_get_para)
    float(sp.re(poles[0]))

    expn, etal, etar, etaa = [], [], [], []
    poles_allplane = np.array([])
    for i in poles:
        i = complex(i)
        if i.imag < 0:
            expn.append(i * 1.J)
        poles_allplane = np.append(poles_allplane, i)

    expn = np.array(expn)

    expn_imag_sort = np.argsort(np.abs(np.imag(expn)))[::-1]
    expn_imag = np.sort(np.abs(np.imag(expn)))[::-1]
    expn_val_cc = expn[expn_imag_sort[expn_imag != 0]]
    expn_val_n_cc = expn[expn_imag_sort[expn_imag == 0]]

    for ii in range(0, len(expn_val_cc), 2):
        etal.append(
            complex(
                sp.N((-1.j * numer_get_para /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii]}))))

        etal.append(
            complex(
                sp.N((-1.j * numer_get_para /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii + 1]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii + 1]}))))

        etar.append(np.conj(etal[-1]))
        etar.append(np.conj(etal[-2]))
        etaa.append(np.sqrt(np.abs(etal[-2]) * np.abs(etar[-2])))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    for ii in range(len(expn_val_n_cc)):
        etal.append(
            complex(
                sp.N((-1.j * numer_get_para /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_n_cc[ii]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_n_cc[ii]}))))
        etar.append(np.conj(etal[-1]))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    return np.array(etal), np.array(
        etar), np.array(etaa), np.array(expn)


def prony_find_gamma(h, n_sample, nind):
    mat_h = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        mat_h[i, :] = h[i:n_sample + i]
    sing_vs, Q = LA.eig(mat_h)
    phase_mat = np.diag([np.exp(-1j * np.angle(sing_v) / 2.0)
                        for sing_v in sing_vs])
    vs = np.array([np.abs(sing_v) for sing_v in sing_vs])
    Qp = np.dot(Q, phase_mat)
    sort_array = np.argsort(vs)[::-1]
    vs = vs[sort_array]
    Qp = (Qp[:, sort_array])

    for i in [nind]:
        print(i)
        gamma = np.roots(Qp[:, i][::-1])
    gamma_new = gamma[np.argsort(np.abs(gamma))[:nind]]
    return gamma_new


def decompose_spe_prony(spe: sp.core.mul.Mul, w_sp: sp.core.symbol.Symbol, sp_para_dict: dict, para_dict: dict, condition_dict: dict, nind: int or list, npsd=100, scale=1, n=1000, bose_fermi=1):
    '''
    decompose the spectrum with prony method
    input
    spe: the spectrum, a sp.core.mul.Mul object (sympy expression)
    w_sp: the frequency symbol
    nind: int or list. If int, find gamma using the real part. If list, then int stand for the number used to find gamma, and character  'a' meaning from the analytical expression.
    '''
    etal_pade, _, _, expn_pade = decompose_spe(
        spe, w_sp, sp_para_dict, para_dict, condition_dict, npsd, bose_fermi=bose_fermi)

    t = np.linspace(0, 1, 2 * n + 1)
    res_t = np.zeros(len(t), dtype=complex)
    fit_t(scale * t, res_t, expn_pade, etal_pade)

    print("check the sample points")
    print(res_t[:10])
    print(res_t[-10:])
    if type(nind) is list:
        if nind[0] == 'a':
            _, _, _, expn_real = decompose_spe_real(spe, w_sp, sp_para_dict, para_dict,
                                                    condition_dict)
            gamma_real = np.exp(- expn_real * scale / (2*n))
            nind[0] = len(gamma_real)
            if bose_fermi == 1:
                print("For the bose case, C(t) have the analytical imag part")
                sys.exit()
            return prony_fitting(res_t, t, nind, scale, n, gamma_real=gamma_real)
        elif nind[1] == 'a':
            _, _, _, expn_imag = decompose_spe_imag(spe, w_sp, sp_para_dict, para_dict,
                                                    condition_dict)
            gamma_imag = np.exp(- expn_imag * scale / (2*n))
            nind[1] = len(gamma_imag)
            if bose_fermi == 2:
                print("For the fermi case, C(t) have the analytical real part")
                sys.exit()
            return prony_fitting(res_t, t, nind, scale, n, gamma_imag=gamma_imag)
    return prony_fitting(res_t, t, nind, scale, n)


def decompose_spe_prony_na(spe: sp.core.mul.Mul, w_sp: sp.core.symbol.Symbol, sp_para_dict: dict, para_dict: dict, condition_dict: dict, nind: int or list, scale=1, n=1000, n_fft=1000000, scale_fft=2000, bose_fermi=1):
    '''
    decompose the non-analytic spectrum with prony method
    input
    spe: the spectrum, a sp.core.mul.Mul object (sympy expression)
    w_sp: the frequency symbol
    nind: int or list.
        If int, find gamma using the real part.
        If list of two int, then each int value stand for the number used to find gamma.
        Character  'a' meaning from the analytic expression.
    '''
    n_rate = int(scale_fft * scale / (4 * n))
    print("this should be any int value: ", scale_fft * scale / (4 * n))

    beta = para_dict['beta']
    gen_jw = sp.lambdify(w_sp, spe.subs(sp_para_dict))

    w = np.linspace(0, scale_fft * np.pi, n_fft + 1)[:-1]
    dw = w[1] - w[0]
    jw1 = gen_jw(w)
    jw2 = gen_jw(-w)

    if (bose_fermi == 1):
        cw1 = jw1 / (1 - np.exp(-beta * w))
        cw2 = jw2 * np.exp(-beta * w) / (1 - np.exp(-beta * w))
    cw1[0] = cw1[1] / 2
    cw2[0] = cw2[1] / 2
    del jw1, jw2

    print("check the sample points")
    print(cw1[:10])
    print(cw1[-10:])
    print(cw2[:10])
    print(cw2[-10:])

    fft_ct = (np.fft.fft(cw1) * dw - np.fft.ifft(cw2) * len(cw2) * dw) / np.pi
    fft_t = 2 * np.pi * np.fft.fftfreq(len(cw1), dw)
    del cw1, cw2

    fft_ct = fft_ct[(scale >= fft_t) & (fft_t >= 0)][::n_rate]
    fft_t = fft_t[(scale >= fft_t) & (fft_t >= 0)][::n_rate]
    print("check the sample points")
    print(fft_ct[:10])
    print(fft_ct[-10:])
    return prony_fitting(fft_ct, fft_t, nind, scale, n)


def prony_fitting(res_t, t, nind, scale, n, gamma_real=None, gamma_imag=None):
    n_sample = (len(t) + 1) // 2
    h = res_t
    if type(nind) is list:
        if (gamma_real is None):
            gamma_real = prony_find_gamma(np.real(h), n_sample, nind[0])
        else:
            gamma_real = np.array(gamma_real)
        if (gamma_imag is None):
            gamma_imag = prony_find_gamma(np.imag(h), n_sample, nind[1])
        else:
            gamma_imag = np.array(gamma_imag)
        gamma = np.append(gamma_real, gamma_imag)
        n_row = nind[0] + nind[1]
    else:
        gamma = prony_find_gamma(np.real(h), n_sample, nind)
        n_row = nind

    t_new = 2*n*np.log(gamma)
    n_col = n_sample*2-1
    gamma_m = np.zeros((2 * n_col, 2 * n_row), dtype=float)
    for i in range(n_row):
        for j in range(n_col):
            gamma_m[j, i] = np.real(gamma[i]**j)
            gamma_m[n_col + j, n_row + i] = np.real(gamma[i]**j)
            gamma_m[j, n_row + i] = -np.imag(gamma[i]**j)
            gamma_m[n_col + j, i] = np.imag(gamma[i]**j)
    h_m = np.append(np.real(h), np.imag(h))

    freq_m = np.zeros((2 * n_col, 2 * n_row), dtype=float)

    C = numpy_to_cvxopt_matrix(gamma_m)
    d = numpy_to_cvxopt_matrix(h_m)
    A = numpy_to_cvxopt_matrix(-freq_m)
    b = numpy_to_cvxopt_matrix(np.zeros(2 * n_col))
    Q = C.T * C
    q = - d.T * C

    opts = {'show_progress': False, 'abstol': 1e-24,
            'reltol': 1e-24, 'feastol': 1e-24}
    for k, v in opts.items():
        solvers.options[k] = v

    sol = solvers.qp(Q, q.T, A, b, None, None, None, None)
    omega_new_temp = np.array(sol['x']).reshape(2, n_row)
    omega_new = omega_new_temp[0, :] + 1.j*omega_new_temp[1, :]

    etal_p = omega_new
    expn_p = -t_new / scale
    return sort_symmetry(etal_p, expn_p)


def benchmark(file_str, magic_str, dir_str='bose', if_np=True, if_save=False):
    cmd = r'mv {} {}-{}'.format(file_str, file_str, magic_str)
    result = subprocess.call(cmd, shell=True)
    if os.path.exists('result/{}/{}-{}.npy'.format(dir_str, file_str, magic_str)):
        data1 = np.load(
            'result/{}/{}-{}.npy'.format(dir_str, file_str, magic_str))
    else:
        data1 = fastread_np(
            'result/{}/{}-{}'.format(dir_str, file_str, magic_str))
    data2 = fastread_np('./{}-{}'.format(file_str, magic_str))
    if if_save:
        np.save('result/{}/{}-{}'.format(dir_str, file_str, magic_str), data2)
    result = (np.sum(np.abs(data1 - data2)))
    if float(result) > 1e-6:
        print(result, 'FAILED')
        print(result, '!!!!!!!!')
    else:
        print(result, 'PASSED')


def thermal_equilibrium(beta, hams):
    return matrix_exp(- beta * hams) / np.trace(matrix_exp(- beta * hams))


def direct_product(a, *args):
    c = a.copy()
    for arg in args:
        if not isinstance(arg, np.ndarray):
            raise TypeError('Input must be numpy.ndarray')
        c = direct_product_2d(c, arg)
    return c


def decompose_spe(spe, w_sp, sp_para_dict, para_dict, condition_dict, npsd, pade=1, bose_fermi=1):
    if (sp.cancel(
            spe.subs(condition_dict)).as_real_imag()[1] == 0):
        imag_part = sp.cancel(
            spe.subs(condition_dict)).as_real_imag()[0]
    else:
        imag_part = sp.cancel(
            spe.subs(condition_dict)).as_real_imag()[1]
    numer, denom = sp.cancel(sp.factor(imag_part)).as_numer_denom()
    numer_get_para = (sp.factor(numer)).subs(sp_para_dict)
    denom_get_para = (sp.factor(denom)).subs(sp_para_dict)

    poles = sp.nroots(denom_get_para)
    float(sp.re(poles[0]))

    expn = []
    poles_allplane = np.array([])
    for i in poles:
        i = complex(i)
        if i.imag < 0:
            expn.append(i * 1.J)
        poles_allplane = np.append(poles_allplane, i)

    etal = []
    etar = []
    etaa = []

    expn = np.array(expn)

    expn_imag_sort = np.argsort(np.abs(np.imag(expn)))[::-1]
    expn_imag = np.sort(np.abs(np.imag(expn)))[::-1]

    expn_val_cc = expn[expn_imag_sort[expn_imag != 0]]
    expn_val_n_cc = expn[expn_imag_sort[expn_imag == 0]]


    expn = list(expn[expn_imag_sort])
    pole, resi = PSD(npsd, bose_fermi, pade)
    beta = para_dict['beta']
    temp = 1 / beta

    for ii in range(0, len(expn_val_cc), 2):
        etal.append(
            complex(
                sp.N((-2.j * numer_get_para /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii]}) *
                     fBose(-1.J * expn_val_cc[ii] / temp, pole, resi))))

        etal.append(
            complex(
                sp.N((-2.j * numer_get_para /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_cc[ii + 1]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_cc[ii + 1]}) *
                     fBose(-1.J * expn_val_cc[ii + 1] / temp, pole, resi))))

        etar.append(np.conj(etal[-1]))
        etar.append(np.conj(etal[-2]))
        etaa.append(np.sqrt(np.abs(etal[-2]) * np.abs(etar[-2])))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    for ii in range(len(expn_val_n_cc)):
        etal.append(
            complex(
                sp.N((-2.j * numer_get_para /
                      np.multiply.reduce(w_sp - poles_allplane[np.abs(
                          poles_allplane + 1.J * expn_val_n_cc[ii]) > 1e-14])
                      ).subs({w_sp: -1.j * expn_val_n_cc[ii]}) *
                     fBose(-1.J * expn_val_n_cc[ii] / temp, pole, resi))))
        etar.append(np.conj(etal[-1]))
        etaa.append(np.sqrt(np.abs(etal[-1]) * np.abs(etar[-1])))

    f = numer_get_para / np.multiply.reduce(w_sp - poles_allplane)
    f = sp.lambdify(w_sp, f)

    for inma in range(len(pole)):
        zomg = -1.J * pole[inma] * temp
        jsum = np.sum(f(zomg))
        expn.append(pole[inma] * temp)
        etal.append(-2.J * resi[inma] * temp * jsum)
        etar.append(np.conj(etal[-1]))
        etaa.append(np.abs(etal[-1]))

    etal = np.array(etal)
    etar = np.array(etar)
    etaa = np.array(etaa)
    expn = np.array(expn)
    return etal, etar, etaa, expn


def single_oscillator(omega, beta):
    etal = np.array([1/(2 * (1-np.exp(-beta * omega))), -1 /
                     (2 * (1-np.exp(beta * omega)))], dtype=complex)
    etar = np.array([-1/(2 * (1-np.exp(beta * omega))), 1 /
                     (2 * (1-np.exp(-beta * omega)))], dtype=complex)
    etaa = np.sqrt(np.abs(etal + etar))
    expn = np.array([1.j * omega, -1.j * omega])
    return etal, etar, etaa, expn


def fastread_np(str):
    return fastread(str).to_numpy()


def sum_exptontial_spectrum(w, res, expn, etal, sigma):
    for i in range(len(etal)):
        res += etal[i] / (expn[i] + sigma * 1.j * w)


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    elif isinstance(o, np.int32):
        return int(o)
    raise TypeError


def sort_symmetry(etal, expn, if_sqrt=True):
    expn_imag_sort = np.argsort(np.abs(np.imag(expn)))[::-1]
    expn_imag = np.sort(np.abs(np.imag(expn)))[::-1]
    expn = expn[expn_imag_sort]
    etal = etal[expn_imag_sort]
    etar = etal[expn_imag_sort]
    expn_val_cc = np.where(expn[expn_imag > 1e-10])[0]
    etaa = np.zeros(len(etal), dtype=float)
    for ii in range(0, len(expn_val_cc), 2):
        even_i = ii
        odd_i = ii + 1
        etar[even_i] = np.conj(etal[odd_i])
        etar[odd_i] = np.conj(etal[even_i])
        etaa[even_i] = np.abs(etal[even_i])
        etaa[odd_i] = np.abs(etal[odd_i])
    for ii in range(len(expn_val_cc), len(expn)):
        even_i = ii
        etar[even_i] = np.conj(etal[even_i])
        etaa[even_i] = np.abs(etal[even_i])
    if (if_sqrt):
        etaa = np.sqrt(etaa)
    print("changelog: change expn, etal, etar, etaa to etal, etar, etaa, expn")
    return etal, etar, etaa, expn


def complex_2_json(list_input, if_dense=None):
    '''
    Convert a complex matrix to a Json format.

    Parameters
    ----------
    if_dense: False, True or None.
        If the input is a dense matrix. If None, then the function will try to determine whether the input is a dense matrix or not.

    Returns
    -------
    json_init: dict
    '''
    if (if_dense is None):
        if len(np.shape(list_input)) > 1:
            index_list = np.where(abs(list_input.flatten()) > 1e-10)
            if (5 * len(index_list) > len(list_input.flatten())):
                if_dense = True
            else:
                if_dense = False
        else:
            if_dense = True

    if (type(list_input) == np.ndarray):
        if (if_dense):
            return {
                "if_initial": True,
                "real": list(np.real(list_input.flatten())),
                "imag": list(np.imag(list_input.flatten()))
            }
        else:
            index_list = np.where(abs(list_input) > 1e-10)
            json_init = {
                "if_initial": True,
                "if_dense": False,
                "length": len(index_list[-1]),
                "i": list(index_list[-2]),
                "j": list(index_list[-1]),
                "real": list(np.real(list_input[index_list])),
                "imag": list(np.imag(list_input[index_list]))
            }
            if len(index_list) > 2:
                json_init["k"] = list(index_list[-3])
            return json_init
    else:
        return {
            "real": np.real(list_input),
            "imag": np.imag(list_input)
        }

def compute_qmd(qmd1a, qmd1c, mode, nsys, etaa, etal, etar):
    qmdta_l = np.zeros((len(mode), nsys, nsys), dtype=complex)
    qmdta_r = np.zeros((len(mode), nsys, nsys), dtype=complex)
    qmdtc_l = np.zeros((len(mode), nsys, nsys), dtype=complex)
    qmdtc_r = np.zeros((len(mode), nsys, nsys), dtype=complex)
    for i in range(len(mode)):
        i_mod = mode[i]
        qmdta_l[i, :, :] = qmd1a[i_mod, :, :] * np.sqrt(etaa[i])
        qmdta_r[i, :, :] = qmd1a[i_mod, :, :] * np.sqrt(etaa[i])
        qmdtc_l[i, :, :] = qmd1c[i_mod, :, :] * etal[i] / np.sqrt(etaa[i])
        qmdtc_r[i, :, :] = qmd1c[i_mod, :, :] * etar[i] / np.sqrt(etaa[i])
    return qmdta_l, qmdta_r, qmdtc_l, qmdtc_r



def init_qmd(json_init, qmd1a, qmd1c, mode, nsys, etaa, etal, etar, if_dense=None):
    qmdta_l, qmdta_r, qmdtc_l, qmdtc_r = compute_qmd(qmd1a, qmd1c, mode, nsys, etaa, etal, etar)
    json_init["qmdta_l"] = complex_2_json(qmdta_l, if_dense=if_dense)
    json_init["qmdta_r"] = complex_2_json(qmdta_r, if_dense=if_dense)
    json_init["qmdtc_l"] = complex_2_json(qmdtc_l, if_dense=if_dense)
    json_init["qmdtc_r"] = complex_2_json(qmdtc_r, if_dense=if_dense)


# Do some normalize thing, you can find more details in the pdf file.
def init_qmd_quad(json_init, qmd2a, qmd2b, qmd2c, mode, nsys, nind, nmod, etaa, etal, etar, if_dense=None):
    qmdt2a_l = np.zeros((nind*nind, nsys, nsys), dtype=complex)
    qmdt2a_r = np.zeros((nind*nind, nsys, nsys), dtype=complex)
    qmdt2b_l = np.zeros((nind*nind, nsys, nsys), dtype=complex)
    qmdt2b_r = np.zeros((nind*nind, nsys, nsys), dtype=complex)
    qmdt2c_l = np.zeros((nind*nind, nsys, nsys), dtype=complex)
    qmdt2c_r = np.zeros((nind*nind, nsys, nsys), dtype=complex)
    for i in range(len(mode)):
        for j in range(len(mode)):
            i_mod = mode[i]
            j_mod = mode[j]
            index_mat = i * nind + j
            qmdt2a_l[index_mat, :, :] = qmd2a[i_mod, j_mod, :, :] * \
                np.sqrt(etaa[i]) * np.sqrt(etaa[j])
            qmdt2a_r[index_mat, :, :] = qmd2a[i_mod, j_mod, :, :] * \
                np.sqrt(etaa[i]) * np.sqrt(etaa[j])
            qmdt2b_l[index_mat, :, :] = qmd2b[i_mod, j_mod, :, :] * \
                etal[i] / np.sqrt(etaa[i]) * np.sqrt(etaa[j])
            qmdt2b_r[index_mat, :, :] = qmd2b[i_mod, j_mod, :, :] * \
                etar[i] / np.sqrt(etaa[i]) * np.sqrt(etaa[j])
            qmdt2c_l[index_mat, :, :] = qmd2c[i_mod, j_mod, :, :] * \
                etal[i] * etal[j] / np.sqrt(etaa[i]) / np.sqrt(etaa[j])
            qmdt2c_r[index_mat, :, :] = qmd2c[i_mod, j_mod, :, :] * \
                etar[i] * etar[j] / np.sqrt(etaa[i]) / np.sqrt(etaa[j])
    json_init["qmdt2a_l"] = complex_2_json(qmdt2a_l, if_dense=if_dense)
    json_init["qmdt2a_r"] = complex_2_json(qmdt2a_r, if_dense=if_dense)
    json_init["qmdt2b_l"] = complex_2_json(qmdt2b_l, if_dense=if_dense)
    json_init["qmdt2b_r"] = complex_2_json(qmdt2b_r, if_dense=if_dense)
    json_init["qmdt2c_l"] = complex_2_json(qmdt2c_l, if_dense=if_dense)
    json_init["qmdt2c_r"] = complex_2_json(qmdt2c_r, if_dense=if_dense)
