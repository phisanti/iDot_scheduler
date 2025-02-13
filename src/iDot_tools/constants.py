from enum import Enum

# Basic configuration settings
CONFIG = {
    'DEAD_VOLUME': 1,
    'MAX_SHEETS': 4,
    'VALID_PLATE_SIZES': {96, 384, 1536}
}


# Dictionary to convert row id between 1536 to iDot naming:
ROWDICT = {
    "A": "Aa",
    "B": "Ab",
    "C": "Ac",
    "D": "Ad",
    "E": "Ba",
    "F": "Bb",
    "G": "Bc",
    "H": "Bd",
    "I": "Ca",
    "J": "Cb",
    "K": "Cc",
    "L": "Cd",
    "M": "Da",
    "N": "Db",
    "O": "Dc",
    "P": "Dd",
    "Q": "Ea",
    "R": "Eb",
    "S": "Ec",
    "T": "Ed",
    "U": "Fa",
    "V": "Fb",
    "W": "Fc",
    "X": "Fd",
    "Y": "Ga",
    "Z": "Gb",
    "ZA": "Gc",
    "ZB": "Gd",
    "ZC": "Ha",
    "ZD": "Hb",
    "ZE": "Hc",
    "ZF": "Hd"
  }


class ParallelisationType(Enum):
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    NONE = "none"