"""
Forest Light Environmental Simulator (FLiES)
Artificial Neural Network Implementation
for the Breathing Earth Systems Simulator (BESS)
"""
import warnings

from .FLiESANN import *
from .version import __version__

__author__ = "Gregory H. Halverson, Robert Freepartner, Hideki Kobayashi, Youngryel Ryu"

warnings.filterwarnings(
	"ignore",
	message="__array_wrap__ must accept context and return_scalar arguments*",
	category=DeprecationWarning
)
