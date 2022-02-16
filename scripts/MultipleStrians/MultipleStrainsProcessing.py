import csv
import MultipleStrainsParser
import StrainParser
from ElongationRateProximity import ElongationProximity

def MultipleStrains (fileName,interval_time,marker1,marker2):
    #dictionary of results lists
    finalCalculation_dict={}

    ##################################################### start Processing #############################################

    #creat a class object & call object method
    #Goal: I store the values of useful columns of the input file for each bacteria in the constructed object attributes.
    #useful columns: ImageNumber,ObjectNumber,TrackObjects_ParentObjectNumber_50,TrackObjects_ParentImageNumber_50
                    #TrackObjects_Label_50,Location_Center_X,Location_Center_Y
                    #AreaShape_MajorAxisLength
    # Object instantiation
    data = MultipleStrainsParser.ObjectParser()
    #Accessing class attributes and method through data
    #function: read csv file and fetch the values of useful columns of the input file for each bacteria
    #Arguments:
    #           fileName: name of CSV file -> read CSV files with this name
    #           name of fluorescent markers columns in the CSV file
    #           they are used to fetch values of these columns
    #           marker1:  name of target fluorescent marker
    #           marker2:  name of second fluorescent marker          
    data.Parse(fileName,marker1,marker2)


    #finalCalculation: nested list
    #output of ElongationProximity function:
    #elongation_rate list,first_time_step list,last_time_step list,final_lable list,first_length list,
    #last_length list,average_proximity list
    #I store all of output lists into finalCalculation list
    finalCalculation_dict=ElongationProximity(interval_time,data.time_steps,data.all_bac_row_index,data.object_index,data.object_lable,
                                         data.length,data.Parent_index,data.Parent_time_step,data.marker2,data.x_location,data.y_location)

    #write results
    with open('OutputFile/'+fileName.split('/')[-1]+'_final_analysis.csv', 'w',newline='') as myfile:
        wr = csv.writer(myfile)
        #header
        wr.writerow(['lable','first time step','last time step','First Length','Last Length','elongation rate','average proximity'])
        #rows
        for counter,element in enumerate(finalCalculation_dict['first_time_step']):
                  wr.writerow([finalCalculation_dict['final_lable'][counter],element,
                  finalCalculation_dict['last_time_step'][counter],finalCalculation_dict['first_length'][counter],
                  finalCalculation_dict['last_length'][counter],finalCalculation_dict['elongation_rate'][counter],
                  finalCalculation_dict['average_proximity'][counter]])

    print('Calculations were done!')
    print('The calculation file was written in the OutputFile folder!')

    
