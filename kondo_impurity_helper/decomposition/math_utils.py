import numpy as np

from kondo_impurity_helper.decomposition.bath_type import BathType

from typing import Any

def bose_function(w: np.ndarray, beta: float, *args: Any, **kwargs: Any) -> np.ndarray:
    return 1.0 / (1.0 - np.exp(-beta * w))   

def fermi_function(w: np.ndarray, beta: float, sigma: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(beta * w * sigma))

def get_sigma(bath_type: BathType) -> int:
    if (BathType.BOSE == bath_type) or (BathType.FERMI_MINUS == bath_type):
        return -1
    elif BathType.FERMI_PLUS == bath_type:
        return 1
    else:
        raise ValueError(f"Unknown bath type: {bath_type}")
        

def bath_statistical_series(x: complex, statistic_poles: np.ndarray, statistic_residues: np.ndarray, bath_type: BathType) -> complex:
    """Compute the statistical function at a given point x, i.e., x = beta omega."""
    sigma = get_sigma(bath_type)
    series_expansion = 0.5 - 2.0 * sigma * x * np.sum(statistic_residues / (x**2 + statistic_poles**2))
    # return series_expansion
    if bath_type == BathType.BOSE:
        return 1.0 / x + series_expansion 
    else:
        return series_expansion