# %%
import numpy as np
import sympy as sp

from kondo_impurity_helper.decomposition.bath_type import BathType
from kondo_impurity_helper.decomposition.mastubara import matsubara_decomposition
from kondo_impurity_helper.decomposition.pade import pade_decomposition, PadeSchemes
from kondo_impurity_helper.decomposition.math_utils import get_sigma, bath_statistical_series, bose_function, fermi_function

from enum import Enum
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass

class DecompositionType(Enum):
    MASTUBARA = 1
    PADE_N_MINUS_1_N = 2
    PADE_N_N = 3
    PADE_N_PLUS_1_N = 4
    
    
def get_pade_scheme(decomposition_type: DecompositionType) -> PadeSchemes:
    if decomposition_type == DecompositionType.PADE_N_MINUS_1_N:
        return PadeSchemes.N_MINUS_1_N
    elif decomposition_type == DecompositionType.PADE_N_N:
        return PadeSchemes.N_N
    elif decomposition_type == DecompositionType.PADE_N_PLUS_1_N:
        return PadeSchemes.N_PLUS_1_N


def compute_spectrum_poles(spectral_function: sp.Expr, parameter_dict: Dict[sp.Symbol, float]) -> List[float]:
    _, denominator = spectral_function.as_numer_denom()
    return compute_expr_roots(denominator, parameter_dict)

def compute_expr_roots(expr: sp.Expr, parameter_dict: Dict[sp.Symbol, float]) -> List[float]:
    expr_subs = sp.factor(expr).subs(parameter_dict)
    poles = sp.nroots(expr_subs)
    return poles

def get_raw_expn(spectrum_poles: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    expn = np.array([], dtype=np.complex128)
    poles_all_plane = np.array([], dtype=np.complex128)
    for sp in np.array(spectrum_poles, dtype=np.complex128):
        poles_all_plane = np.append(poles_all_plane, sp)
        if np.imag(sp) < 0:
            expn = np.append(expn, sp * 1.0j)
    
    argsort = np.argsort(np.abs(np.imag(expn)))
    argsort = np.flip(argsort)
    expn_imag = np.imag(expn[argsort])
    
    real_expn = expn[expn_imag == 0]
    complex_expn = expn[expn_imag != 0]
    expn = expn[argsort]
    return expn, real_expn, complex_expn, poles_all_plane 

@dataclass
class SpectrumPoles:
    """Poles of the spectral function in the complex plane."""
    expn: np.ndarray
    expn_real: np.ndarray
    expn_complex: np.ndarray
    poles_all_plane: np.ndarray
    
    @classmethod
    def from_unsorted_array(cls, spectrum_poles: List[float]):
        expn, expn_real, expn_complex, poles_all_plane = get_raw_expn(spectrum_poles)
        return cls(expn, expn_real, expn_complex, poles_all_plane)
    
@dataclass
class StatisticPoles:
    """Poles of the statistical function, e.g., Fermi/Bose, in the complex plane."""
    N: int
    bath_type: BathType 
    statistic_poles: np.ndarray
    statistic_residues: np.ndarray
    
    @classmethod
    def from_decomposition(cls, N: int, decomposition_type: DecompositionType, bath_type: BathType):
        if decomposition_type == DecompositionType.MASTUBARA:
            statistic_poles, statistic_residues = matsubara_decomposition(N, bath_type)
        else:
            statistic_poles, statistic_residues = pade_decomposition(N, bath_type, get_pade_scheme(decomposition_type)) 
        return cls(N, bath_type, statistic_poles, statistic_residues)
    
@dataclass
class SpectrumFunction:
    omega: sp.Symbol
    expr: sp.Expr 
    numerator: sp.Expr
    denominator: sp.Expr
    parameter_dict: Dict[sp.Symbol, float]
    
    @classmethod
    def from_expr(cls, omega: sp.Symbol, expr: sp.Expr, parameter_dict: Dict[sp.Symbol, float]):
        numerator, denominator = expr.as_numer_denom()
        return cls(omega, expr, numerator, denominator, parameter_dict)
    
def compute_one_spectrum_function_pole(beta: float, omega: float, numerator_function: Callable[[complex], complex], poles_all_plane: np.ndarray, statistic_poles: StatisticPoles):
    TOL = 1e-14
    sigma = get_sigma(statistic_poles.bath_type)
    
    pole = sigma * 1.0j * omega
    
    mask = np.where(np.abs(poles_all_plane - pole) > TOL)[0]
    other_poles = poles_all_plane[mask] 
    
    numerator_val = 2.0j * sigma * numerator_function(pole)
    denumerator_val = np.prod(pole - other_poles)
    statistic_series = bath_statistical_series(x=beta*pole, statistic_poles=statistic_poles.statistic_poles, statistic_residues=statistic_poles.statistic_residues, bath_type=statistic_poles.bath_type)
    return numerator_val / denumerator_val * statistic_series

def compute_one_statistic_function_pole(beta: float, pole: complex, residule: complex, bath_type: BathType, numerator_function: Callable[[complex], complex], poles_all_plane: np.ndarray):
    sigma = get_sigma(bath_type)
    zomega = 1.0j * pole * sigma / beta
    
    numerator = -2.0j / beta * numerator_function(zomega)
    denumerator = np.prod(zomega - poles_all_plane)
    
    return numerator / denumerator * residule
    

def sum_over_poles(beta: float, spectrum_poles: SpectrumPoles, statistic_poles: StatisticPoles, spectrum_function: SpectrumFunction):
    expn = spectrum_poles.expn
    etal = np.array([], dtype=np.complex128)
    etar = np.array([], dtype=np.complex128)
    etaa = np.array([], dtype=np.complex128)
    
    real_expn, complex_expn = spectrum_poles.expn_real, spectrum_poles.expn_complex
    
    numerator_subs = spectrum_function.numerator.subs(spectrum_function.parameter_dict)
    numerator_function = sp.lambdify(spectrum_function.omega, numerator_subs, 'numpy')
    
    # compute the real poles from the spectral function
    for ii in range(0, len(complex_expn), 2):
        val1 = compute_one_spectrum_function_pole(beta, complex_expn[ii], numerator_function, spectrum_poles.poles_all_plane, statistic_poles)
        val2 = compute_one_spectrum_function_pole(beta, complex_expn[ii + 1], numerator_function, spectrum_poles.poles_all_plane, statistic_poles)
        etal = np.append(etal, [val1, val2])
        etar = np.append(etar, [val2.conjugate(), val1.conjugate()])
        absval = np.sqrt(np.abs(val1) * np.abs(val2))
        etaa = np.append(etaa, [absval, absval])
        
    # compute the purely imaginary poles from the spectral function
    for jj in range(len(real_expn)):
        val = compute_one_spectrum_function_pole(beta, real_expn[jj], numerator_function, spectrum_poles.poles_all_plane, statistic_poles)
        etal = np.append(etal, val)
        etar = np.append(etar, val.conjugate())
        etaa = np.append(etaa, np.abs(val))
        
    # compute the poles from the statistical function
    for (pole, residule) in zip(statistic_poles.statistic_poles, statistic_poles.statistic_residues):
        val = compute_one_statistic_function_pole(beta, pole, residule, statistic_poles.bath_type, numerator_function, spectrum_poles.poles_all_plane)
        
        expn = np.append(expn, pole / beta)
        etal = np.append(etal, val)
        etar = np.append(etar, val.conjugate())
        etaa = np.append(etaa, np.abs(val))
    
    return expn, etal, etar, etaa

def decompose_spectrum(omega: sp.Symbol, spectral_function: sp.Expr, parameter_dict: Dict[sp.Symbol, float], N: int, beta: float, decomposition_type: DecompositionType, bath_type: BathType):
    spec_function = SpectrumFunction.from_expr(omega, spectral_function, parameter_dict)
    statistic_poles = StatisticPoles.from_decomposition(N, decomposition_type, bath_type)
    raw_spec_poles = compute_spectrum_poles(spectral_function, parameter_dict)
    spec_poles = SpectrumPoles.from_unsorted_array(raw_spec_poles)
    
    return sum_over_poles(beta, spec_poles, statistic_poles, spec_function)

def correlation_function(w, expn, etal, sigma):
    res = np.zeros(w.shape, dtype=np.complex128)
    for i in range(len(etal)):
        res[:] += etal[i] / (expn[i] + sigma * 1.j * w)
    return res

def main_bose():
    # from zihao import decompose_spe
    
    # --- Fermi spectral function ---
    # Delta, W, omega= sp.symbols('Delta W omega')
    # spectral_function = Delta * W / (W**2 + omega**2)
    # parameter_dict = {Delta: 0.3, W: 1.04}
    
    # --- Bose spectral function ---
    lambd, omega, eta = sp.symbols('lambda omega eta', real=True)
    spectral_function_complex = 2 * lambd * eta / (eta - sp.I * omega)
    re, im = sp.factor(spectral_function_complex).as_real_imag()
    spectral_function = 2 * lambd * omega * eta / (omega**2 + eta**2)
    parameter_dict = {lambd: 0.01, eta: 0.4}
    
    
    
    decomposition_type = DecompositionType.PADE_N_MINUS_1_N
    # bath_type = BathType.FERMI_PLUS
    bath_type = BathType.BOSE
    N = 2
    
    beta = 0.323
    expn, etal, etar, etaa = decompose_spectrum(omega, spectral_function, parameter_dict, N, beta, decomposition_type, bath_type)
    
    from deom import decompose_spe
    
    etal_deom, etar_deom, etaa_deom, expn_deom = decompose_spe(spectral_function_complex, omega, parameter_dict, {'beta': beta}, {}, N)
    
    L = 30
    w = np.linspace(-L, L, 1000)
    
    jw_func = sp.lambdify(omega, spectral_function.subs(parameter_dict), 'numpy')
     
    expected = bose_function(w, beta) * jw_func(w)
    jwfw = correlation_function(w, expn, etal, get_sigma(bath_type))
    jwfw_deom = correlation_function(w, expn_deom, etal_deom, get_sigma(bath_type))
    
    from matplotlib import pyplot as plt 
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(w, expected, label='Expected')
    ax.plot(w, jwfw.real, ls='--', label='home-made RE')
    ax.plot(w, jwfw_deom.real, ls=':', label='DEOM RE')
    # ax.plot(w, jwfw_deom.imag, label='DEOM IM')
    
    
    ax.legend()
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$J(\omega)f(\omega)$')
    plt.show()
    
    return expn, etal, etar, etaa

def main_fermi():
    # --- Fermi spectral function ---
    Delta, W, omega= sp.symbols('Delta W omega', real=True)
    spectral_function = Delta * W / (W**2 + omega**2)
    parameter_dict = {Delta: 0.3, W: 1.04} 
    
    decomposition_type = DecompositionType.PADE_N_MINUS_1_N
    bath_type = BathType.FERMI_MINUS
    
    N = 2
    beta = 0.323
    
    expn, etal, etar, etaa = decompose_spectrum(omega, spectral_function, parameter_dict, N, beta, decomposition_type, bath_type)
    
    L = 30
    w = np.linspace(-L, L, 1000)
    
    jw_func = sp.lambdify(omega, spectral_function.subs(parameter_dict), 'numpy')
     
    expected = fermi_function(w, beta, get_sigma(bath_type)) * jw_func(w)
    jwfw = correlation_function(w, expn, etal, get_sigma(bath_type))
    
    from matplotlib import pyplot as plt
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(w, expected, label='Expected')
    ax.plot(w, jwfw.real, ls='--', label='home-made RE')
    # ax.plot(w, jwfw.imag, label='home-made IM')
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$J(\omega)f(\omega)$')
    ax.legend()
    plt.show()
    return expn, etal, etar, etaa
    
    
    
    
    
    
    
    

# %%
if __name__ == '__main__':
    expn, etal, etar, etaa = main_bose()
    expn, etal, etar, etaa = main_fermi()
    
    
    # print(f"{expn=}")
    # print(f"{etal=}")
    # print(f"{etar=}")
    # print(f"{etaa=}")
    
# %%
