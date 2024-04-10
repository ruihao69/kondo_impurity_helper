# %%
import numpy as np
import scipy.linalg as LA

from enum import Enum, unique
from kondo_impurity_helper.decomposition.bath_type import BathType

from typing import Tuple

@unique
class PadeSchemes(Enum):
    N_MINUS_1_N = 1
    N_N = 2
    N_PLUS_1_N = 3
    
def get_bm(m: int, bath_type: BathType) -> int:
    # return 2 * m - 1 if bath_type == BathType.FERMI else 2 * m + 1
    if bath_type == BathType.BOSE:
        return 2 * m + 1
    elif (bath_type == BathType.FERMI_PLUS) or (bath_type == BathType.FERMI_MINUS):
        return 2 * m - 1
    else:
        raise ValueError("Invalid Bath Type")

def get_dm(m: int, bath_type: BathType) -> int:
    if m < 1:
        raise ValueError("Invalid value for m")
    elif m == 1:
        bm = get_bm(m, bath_type)
        return 1 / (4 * bm)
    elif m % 2 == 0: 
        _m = m // 2 
        b_m = get_bm(_m, bath_type)
        b_2m = get_bm(2*_m, bath_type)
        return -4 * _m**2 * b_m**2 * b_2m
    else:
        _m = (m - 1) // 2
        b_m = get_bm(_m, bath_type)
        b_m_plus_1 = get_bm(_m+1, bath_type)
        b_2m_plus_1 = get_bm(2*_m+1, bath_type)
        return - b_2m_plus_1 / (4 * _m * (_m + 1) * b_m * b_m_plus_1)
        # return - get_bm(2*_m+1, bath_type) / (4 * _m * (_m + 1) * get_bm(_m, bath_type) * get_bm(_m+1, bath_type))

def get_M(N :int, pade_scheme: PadeSchemes) -> int:
    if pade_scheme == PadeSchemes.N_MINUS_1_N:
        return 2 * N
    elif pade_scheme == PadeSchemes.N_N:
        return 2 * N + 1
    elif pade_scheme == PadeSchemes.N_PLUS_1_N:
        return 2 * N + 2
    else:
        raise ValueError("Invalid Pade Scheme")
    
def get_Lambda_matrix_Q(M: int, bath_type: BathType) -> np.ndarray:
    Lambda = np.zeros((M, M), dtype=np.float64)
    for _m in range(M):
        m = _m + 1
        denominator = np.sqrt(get_bm(m, bath_type) * get_bm(m+1, bath_type))
        if _m == M - 1:
            continue 
        Lambda[_m, _m+1] = 1.0 / denominator
        Lambda[_m+1, _m] = 1.0 / denominator
    return Lambda

def get_Lambda_matrix_P(M: int, bath_type: BathType) -> np.ndarray:
    Lambda = np.zeros((M-1, M-1), dtype=np.float64)
    for _m in range(M-1):
        m = _m + 1
        denominator = np.sqrt(get_bm(m+1, bath_type) * get_bm(m+2, bath_type))
        if _m + 1 == M - 1:
            continue    
        Lambda[_m, _m+1] = 1.0 / denominator
        Lambda[_m+1, _m] = 1.0 / denominator
    return Lambda

def get_Lambda_prime_matrix_Q(M: int, bath_type: BathType):
    Lambda = np.zeros((M-1, M-1), dtype=np.float64)
    for _m in range(M-1):
        if _m + 1 == M - 1:
            continue
        m = _m + 1
        denominator = np.sqrt(get_dm(m+1, bath_type) * get_dm(m+2, bath_type))
        Lambda[_m, _m+1] = 1.0 / denominator
        Lambda[_m+1, _m] = 1.0 / denominator
    return Lambda

def get_Lambda_prime_matrix_P(M: int, bath_type: BathType):
    Lambda = np.zeros((M, M), dtype=np.float64)
    for _m in range(M):
        if _m == M - 1:
            continue
        m = _m + 1
        denominator = np.sqrt(get_dm(m, bath_type) * get_dm(m+1, bath_type))
        Lambda[_m, _m+1] = 1.0 / denominator
        Lambda[_m+1, _m] = 1.0 / denominator
    return Lambda

def numerator_eta(jj: int, N: int, xi: np.ndarray, zeta: np.ndarray) -> float:
    numerator = 1.0
    for kk in range(N):
        numerator *= (zeta[kk]**2 - xi[jj]**2)
    return numerator

def numerator_eta_tilde(jj: int, N: int, xi: np.ndarray, zeta: np.ndarray) -> float:
    numerator = 1.0
    for kk in range(N - 1):
        numerator *= (zeta[kk]**2 - xi[jj]**2)
    return numerator

def denominator_eta(jj: int, N: int, xi: np.ndarray) -> float:
    denominator = 1.0
    for kk in range(N):
        if kk == jj:
            continue
        denominator *= (xi[kk]**2 - xi[jj]**2)
    return denominator

    
def get_eta_N_MINUS_1_N(N: int, LambdaQ: np.ndarray, LambdaP: np.ndarray, bath_type: BathType):
    evalsQ, _ = LA.eigh(LambdaQ)
    xi_tilde = 2 / evalsQ[evalsQ.shape[0]//2:]
    
    evalsP, _ = LA.eigh(LambdaP)
    zeta_tilde = 2 / evalsP[(evalsP.shape[0]//2+1):]
    

    b_N_plus_1 = get_bm(N+1, bath_type)
    eta_tilde = np.zeros(N, dtype=np.float64)
    for jj in range(N):
        eta_tilde[jj] = numerator_eta_tilde(jj, N, xi_tilde, zeta_tilde) / denominator_eta(jj, N, xi_tilde)
    
    residue = eta_tilde = 0.5 * N * b_N_plus_1 * eta_tilde
    poles = xi_tilde
    
    # return poles, residue
    return np.flip(poles), np.flip(residue)

def get_eta_N_N(N: int, LambdaQ: np.ndarray, LambdaP: np.ndarray, bath_type: BathType):
    evalsQ, _ = LA.eigh(LambdaQ)    
    xi = 2 / evalsQ[(evalsQ.shape[0]//2+1):]
    
    evalsP, _ = LA.eigh(LambdaP)
    zeta = 2 / evalsP[(evalsP.shape[0]//2):]
    
    RN = 1 / (4 * (N+1) * get_bm(N+1, bath_type))
    
    eta = np.zeros(N, dtype=np.float64)
    for jj in range(N):
        eta[jj] = numerator_eta(jj, N, xi, zeta) / denominator_eta(jj, N, xi)
   
    residue = eta =  0.5 * RN * eta
    poles = xi  
    
    # return poles, residue
    return np.flip(poles), np.flip(residue)

def get_eta_check_recursive(N: int, xi_check: np.ndarray, bath_type: BathType):
    def get_T_check_N(N: int, bath_type: BathType):
        denominator = 4 * sum(get_dm(m=(_n + 1) * 2, bath_type=bath_type) for _n in range(N + 1))
        try:
            return 1 / denominator
        except ZeroDivisionError:
            message = "Encountered ZeroDivisionError for parameters: "
            message += f"{N=}, {denominator=}"
            raise ZeroDivisionError(message)
        
    def get_t_k(k: int, bath_type: BathType):
        return get_T_check_N(0, bath_type) if k == 1 else get_T_check_N(k-1, bath_type) / get_T_check_N(k-2, bath_type)
    
    def get_delta_j_k(j: int, k: int, xi_check: np.ndarray):
        try:
            return 1 if (k == N+1) or (k == j+1) else xi_check[k-1] ** 2 - xi_check[j] ** 2
        except IndexError:
            message = "Encountered IndexError for parameters: "
            message += f"{j=}, {k=}, {xi_check.shape=}"
            raise IndexError(message)
        
    def get_r_m(m: int, j: int, bath_type: BathType):
        if m == 0:
            return 0
        elif m % 2 == 0:
            k = m // 2
            tk = get_t_k(k, bath_type)
            delta_k = get_delta_j_k(j=j, k=k, xi_check=xi_check)
            sign = np.sign(tk/delta_k)  
            return get_r_m(m-1, j, bath_type) * sign
        else:
            k = (m + 1) // 2
            tk = get_t_k(k, bath_type)
            delta_k = get_delta_j_k(j=j, k=k, xi_check=xi_check)
            return 2 * np.sqrt(np.abs(tk/delta_k))
    
    def get_X_j(X_m_minus_1: float, X_m_minus_2: float, m: int, j: int):
        r_m = get_r_m(m, j, bath_type)
        r_m_minus_1 = get_r_m(m-1, j, bath_type)
        d_m = get_dm(m, bath_type)
        X_m = d_m * r_m * X_m_minus_1 - r_m * r_m_minus_1 * xi_check[j]**2 * X_m_minus_2 / 4
        return X_m, X_m_minus_1
    
    X_neg_1 = 0.0
    X_0 = 0.5
    
    eta_check = np.zeros(N, dtype=np.float64)
    for jj in range(N):
        X_m, X_m_minus_1 = X_0, X_neg_1
        m = 1
        while m <= 2 * N + 2:
            X_m, X_m_minus_1 = get_X_j(X_m, X_m_minus_1, m, jj)
            m += 1
        eta_check[jj] = X_m
        
    poles = xi_check
    residue = eta_check 
    return np.flip(poles), np.flip(residue)

def get_eta_N_PLUS_1_N(N: int, LambdaQ: np.ndarray, bath_type: BathType):
    evalsQ, _ = LA.eigh(LambdaQ)
    xi_check = 2 / evalsQ[(evalsQ.shape[0]//2+1):]
    
    return get_eta_check_recursive(N, xi_check, bath_type)
    
    
def pade_decomposition(N: int, bath_type: BathType, pade_scheme: PadeSchemes = PadeSchemes.N_MINUS_1_N) -> Tuple[np.ndarray, np.ndarray]:
    M = get_M(N, pade_scheme)
    if pade_scheme == PadeSchemes.N_MINUS_1_N:
        LambdaQ = get_Lambda_matrix_Q(M, bath_type)
        LambdaP = get_Lambda_matrix_P(M, bath_type)
        return get_eta_N_MINUS_1_N(N, LambdaQ, LambdaP, bath_type)
    elif pade_scheme == PadeSchemes.N_N:
        LambdaQ = get_Lambda_matrix_Q(M, bath_type)
        LambdaP = get_Lambda_matrix_P(M, bath_type)
        return get_eta_N_N(N, LambdaQ, LambdaP, bath_type)
    elif pade_scheme == PadeSchemes.N_PLUS_1_N:
        LambdaQ = get_Lambda_prime_matrix_Q(M, bath_type)
        return get_eta_N_PLUS_1_N(N, LambdaQ, bath_type)
    else:
        raise ValueError("Invalid Pade Scheme")
    

# %%
if __name__ == "__main__":
    from kondo_impurity_helper.decomposition.mastubara import matsubara_decomposition
    N = 5
    # calculation of poles and residues using N_MINUS_1_N scheme
    bath_type = BathType.FERMI_PLUS
    pade_scheme = PadeSchemes.N_MINUS_1_N
    p, r = pade_decomposition(N, bath_type, pade_scheme)
    print(f"poles:\n {p}")    
    print(f"residues:\n {r}")
    
    # calculation of poles and residues using N_N scheme
    pade_scheme = PadeSchemes.N_N
    p, r = pade_decomposition(N, bath_type, pade_scheme)
    
    print(f"poles:\n {p}")
    print(f"residues:\n {r}")
    
    # calculation of poles and residues using N_PLUS_1_N scheme
    pade_scheme = PadeSchemes.N_PLUS_1_N
    p, r = pade_decomposition(N, bath_type, pade_scheme)
    print(f"poles:\n {p}")
    print(f"residues:\n {r}")
    
    p, r = matsubara_decomposition(N, bath_type)
    print(f"poles:\n {p}")
    print(f"residues:\n {r}")
# %%
