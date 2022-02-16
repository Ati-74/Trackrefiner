import sys
sys.path.append('scripts/merge')
from MergeStrains import MergeTwoStrains


####################################################### main  #######################################################
# get input files (should be in CSV format)
object_file1='C:/Users/Ati/Documents/PipeLineForCPAnalysis/YFP'
object_file2='C:/Users/Ati/Documents/PipeLineForCPAnalysis/RFP'

#first fluorescent marker
color_num1='YFP'
#second fluorescent marker
color_num2='RFP'

#Merge Two Strains files
MergeTwoStrains(object_file1,object_file2,color_num1,color_num2)
