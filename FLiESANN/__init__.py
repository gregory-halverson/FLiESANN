"""
Forest Light Environmental Simulator (FLiES)
Artificial Neural Network Implementation
for the Breathing Earth Systems Simulator (BESS)
"""

from .FLiESANN import *

from os.path import join, abspath, dirname

with open(join(abspath(dirname(__file__)), "version.txt")) as f:
    version = f.read()

__version__ = version
__author__ = "Gregory H. Halverson, Robert Freepartner, Hideki Kobayashi, Youngryel Ryu"
