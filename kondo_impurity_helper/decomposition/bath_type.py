from enum import Enum, unique

@unique
class BathType(Enum):
    BOSE = 1
    FERMI_PLUS = 2
    FERMI_MINUS = 3
    
    