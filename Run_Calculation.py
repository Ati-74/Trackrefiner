import sys
sys.path.append('scripts/MultipleStrians')
sys.path.append('scripts/strain')

from MultipleStrainsProcessing import MultipleStrains
from StrainProcessing import Strain


#Switch between MultipleStrains and one strain stage (if you have only one strain: Multiple_Strains=False)
Multiple_Strains=True
#getting the name of input file (should be in CSV format)
fileName='OutputFile/merge'
#interval time
interval_time=3

#There are two choices: `linearRegression` , `average` (ln(last length)-ln(first length)/lifehistory)
growthrateMethod="average"


#MultipleStrains
#if you have Multiple strains mode, fill these two variables
##getting the name of fluorescent markers color
#name of target fluorescent marker
marker1='YFP'
#name of second fluorescent marker
marker2='RFP'


####################################################### main  #######################################################
if Multiple_Strains:
    MultipleStrains (fileName,interval_time,marker1,marker2,growthrateMethod)
elif Multiple_Strains==False:
    Strain (fileName,interval_time,growthrateMethod)



