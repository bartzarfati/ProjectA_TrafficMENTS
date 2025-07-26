import sys

from pyRDDLGym import ExampleManager
from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Policies.Agents import RandomAgent


def main() :
    ENV = 'MountainCar_Continuous_gym'
    myEnv = pyRDDLGym.make(ENV, 0)
    
