import numpy as np

from kondo_impurity_helper.decomposition.bath_type import BathType

from typing import Tuple

def matsubara_decomposition(poles_cuoff: int, bath_type: BathType) -> Tuple[np.ndarray, np.ndarray]:
    if BathType.BOSE == bath_type:
        return matsubara_decomposition_bose(poles_cuoff)
    elif BathType.FERMI_PLUS == bath_type or BathType.FERMI_MINUS == bath_type:
        return matsubara_decomposition_fermi(poles_cuoff)
    else:
        raise ValueError(f"Unknown bath type: {bath_type}")


def matsubara_decomposition_bose(poles_cutoff: int) -> Tuple[np.ndarray, np.ndarray]:
    poles = np.array([2 * (i + 1) * np.pi for i in range(poles_cutoff)])
    residue = np.array([1.0 for _ in range(poles_cutoff)])
    return poles, residue


def matsubara_decomposition_fermi(poles_cutoff: int) -> Tuple[np.ndarray, np.ndarray]:
    poles = np.array([np.pi * (2 * i + 1) for i in range(poles_cutoff)])
    residue = np.array([1.0 for _ in range(poles_cutoff)])
    return poles, residue
