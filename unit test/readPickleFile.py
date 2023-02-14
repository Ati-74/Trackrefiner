import pickle
import CellProfilerAnalysis

if __name__ == '__main__':

    file_name = '../doc/step-000089.pickle'
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
    bacterium = cs[197]

    # attributes: 'id', 'label', 'cellType', 'divideFlag', 'cellAge', 'growthRate', 'LifeHistory', 'startVol',
    # 'targetVol', 'pos', 'time', 'radius', 'length', 'dir', 'ends', 'strainRate', 'strainRate_rolling'

    bacterium_id = bacterium.id
    bacterium_growth_rate = bacterium.growthRate
    bacterium_life_history = bacterium.LifeHistory
    bacterium_strainRate_rolling = bacterium.strainRate_rolling
    print('bacteria id: ' + str(bacterium_id))
    print('Growth rate: ' + str(bacterium_growth_rate))
    print('Life History: ' + str(bacterium_life_history))
    print('strainRate_rolling: ' + str(bacterium_strainRate_rolling))
