# %%
import numpy as np
import sympy as sp

from kondo_impurity_helper.decomposition.bath_type import BathType
from kondo_impurity_helper.decomposition.spectrum_decomposition import decompose_spectrum, DecompositionType
from kondo_impurity_helper.decomposition.spectrum_decomposition import correlation_function
from kondo_impurity_helper.decomposition.math_utils import fermi_function, get_sigma
from kondo_impurity_helper.third_party.deom import sort_symmetry

from dataclasses import dataclass

def compute_rmse(
    omega_sp: sp.Symbol,
    spectral_function_expr: sp.Expr,
    parameters_dict: dict,
    beta: float,
    bath_type: BathType,
    expn: np.ndarray,
    etal: np.ndarray,
) -> float:
    L = 30
    omega = np.linspace(-L, L, 3000)
    spectral_function = sp.lambdify(omega_sp, spectral_function_expr.subs(parameters_dict), 'numpy')
    sigma = get_sigma(bath_type)

    expected_integrand = spectral_function(omega) * fermi_function(omega, beta, sigma)
    ffjw = correlation_function(omega, expn, etal, sigma).real
    return np.sqrt(np.mean((expected_integrand - ffjw)**2))


def deconpose_spe_given_tol(
    omega_sp: sp.Symbol,
    spectral_function_expr: sp.Expr,
    parameters_dict: dict,
    beta: float,
    bath_type: BathType,
    tol: float=1e-4
):
    N = 1
    converged = False
    while not converged:
        expn, etal, etar, etaa = decompose_spectrum(
            omega=omega_sp,
            spectral_function=spectral_function_expr,
            parameter_dict=parameters_dict,
            N=N,
            beta=beta,
            decomposition_type=DecompositionType.PADE_N_MINUS_1_N,
            bath_type=bath_type
        )

        rmse = compute_rmse(omega_sp, spectral_function_expr, parameters_dict, beta, bath_type, expn, etal)
        converged = True if rmse < tol else False
        N += 1
    print(f"Converged at N = {N-1} with RMSE = {rmse}")
    return expn, etal, etar, etaa

@dataclass
class NonPolarizedSymmetricFermiBath:
    """ Spin fermion bath functions """
    """ Note that bath is not spin-polarized """
    """ i.e., we assume the spectral functions are the same for both spins """
    """ moreover, we assume that the electrons and holes have identical spectral functions """
    expn: np.ndarray
    etal: np.ndarray
    etar: np.ndarray
    etaa: np.ndarray

    @classmethod
    def from_lorentzian(cls, W: float, beta: float, lambd: float=1.0, rmse_tol: float=1e-4) -> "NonPolarizedSymmetricFermiBath":
        """generate a non-polarized symmetric fermi bath with lorentzian spectral functions. W is the bandwidth, lambd is the scaling factor for density of state per spin, whose unit is 2 / (pi * W).

        Args:
            W (float): The bandwidth of the fermi bath spectral functions.
            lambd (float, optional): scaling factor for density of state per spin. Defaults to 1.0.

        Returns:
            NonPolarizedSymmetricFermiBath: a non-polarized symmetric fermi bath with lorentzian spectral functions.
        """
        # Defaults for the decomposition
        DECOMPOSITION_TYPE = DecompositionType.PADE_N_MINUS_1_N

        omega_sp, W_sp, lambd_sp = sp.symbols('omega W lambd')
        parameters_dict = {W_sp: W, lambd_sp: lambd}
        spectral_function_expr = 2 * lambd_sp * W_sp / (omega_sp**2 + W_sp**2)

        # the partical contributions of the spectral functions
        expn_p, etal_p, etar_p, etaa_p = deconpose_spe_given_tol(
            omega_sp=omega_sp,
            spectral_function_expr=spectral_function_expr,
            parameters_dict=parameters_dict,
            beta=beta,
            bath_type=BathType.FERMI_PLUS,
            tol=rmse_tol
        )

        # the hole contributions of the spectral functions
        expn_h, etal_h, etar_h, etaa_h = deconpose_spe_given_tol(
            omega_sp=omega_sp,
            spectral_function_expr=spectral_function_expr,
            parameters_dict=parameters_dict,
            beta=beta,
            bath_type=BathType.FERMI_MINUS,
            tol=rmse_tol
        )

        # sort the symmetry
        etal_p, etar_p, etaa_p, expn_p = sort_symmetry(etal_p, expn_p, False)
        etal_h, etar_h, etaa_h, expn_h = sort_symmetry(etal_h, expn_h, False)


        expn = np.concatenate([expn_p, expn_p, expn_h, expn_h])
        etal = np.concatenate([etal_p, etal_p, etal_h, etal_h])
        etar = np.concatenate([etar_p, etar_p, etar_h, etar_h])
        etaa = np.concatenate([etaa_p, etaa_p, etaa_h, etaa_h])
        return cls(expn, etal, etar, etaa)

    def get_deom_inputs(self):
        return self.etal, self.etar, self.etaa, self.expn

def main():
    W = 1.0
    beta = 1.0
    bath = NonPolarizedSymmetricFermiBath.from_lorentzian(W, beta)

    etal, etar, etaa, expn = bath.get_deom_inputs()
    #print(etal, etar, etaa, expn)
    #print(f"{len(etal)=}")

# %%
if __name__ == "__main__":
    main()
# %%
