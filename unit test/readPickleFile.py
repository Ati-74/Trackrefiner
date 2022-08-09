import pickle
import sys
sys.path.append('../scripts/strain')
from ProcessCellProfilerData import Dict2Class

file_name = '../doc/step-38.pickle'
data = pickle.load(open(file_name, 'rb'))

# dictionary keys
dictionary_keys = data.keys()
print(dictionary_keys)

# time step
timestep = data['stepNum']
print('time step: ' + str(timestep))

# lineage
lineage = data['lineage']
print('lineage: ' + str(lineage))

# see bacteria information
cs = data['cellStates']

# bacterium
bacterium = cs[1]
# attributes: 'id', 'cellType', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol', 'targetVol', 'pos',
#              'time', 'radius', 'length', 'orientation'

bacterium_id = bacterium.id
bacterium_growth_rate = bacterium.growthRate
bacterium_life_history = bacterium.LifeHistory
print('bacteria id: ' + str(bacterium_id))
print('Growth rate: ' + str(bacterium_growth_rate))
print('Life History: ' + str(bacterium_life_history))
