import sys
import numpy as np
import pickle
import inspect
import CellModeller
import pandas as pd
sys.path.append('../scripts')
from MicroColonyAnalysis import Dict2Class

file_name = '../document/step-56.pickle'
data = pickle.load(open(file_name, 'rb'))

# dictionary keys
dictionary_keys = data.keys()
print(dictionary_keys)

# time step
timestep = data['stepNum']
print('time step: ' + str(timestep))

# number of micro colonies in this time step
number_of_micro_colonies = data['num_micro_colonies']
print('number of micro colonies in this time step: ' + str(number_of_micro_colonies))

# see one micro colony information
cs = data['micro_colonies_States']
# correspond micro colony
correspond_micro_colony = cs[0]
bacteria_id_in_this_micro_colony = correspond_micro_colony.micro_colony_ids
aspect_ratio = correspond_micro_colony.aspect_ratio
anisotropy = correspond_micro_colony.anisotropy
print('bacteria id in this micro colony: ' + str(bacteria_id_in_this_micro_colony))
print('Aspect Ratio: ' + str(aspect_ratio))
print('Anisotropy: ' + str(anisotropy))
    
