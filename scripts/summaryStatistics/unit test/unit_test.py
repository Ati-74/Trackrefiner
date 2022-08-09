import sys

sys.path.append('../scripts')
from MicroColonyAnalysis import process_data

if __name__ == '__main__':
    input_file = '../input files/MyExpt_IdentifySecondaryObjects.csv'
    relationship_file = '../input files/MyExpt_Object relationships.csv'
    output_directory = '../output/'
    summary_statistic_method_list = ["Aspect Ratio", "Anisotropy"]
    process_data(input_file, relationship_file, output_directory, summary_statistic_method_list)
