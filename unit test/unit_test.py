import sys
sys.path.append('../scripts/strain')
from ProcessCellProfilerData import process_data


if __name__ == '__main__':
    input_file = '../examples/SingleStrain/InputFile/single_strain_test.csv'
    output_directory = '../examples/SingleStrain/outputs/'
    process_data(input_file, output_directory)
