import csv
import mergeParser
import mergeProcessing

def MergeTwoStrains(object_file1,object_file2,color_num1,color_num2):
    #Parsing data
    # Object instantiation
    data1 = mergeParser.ObjectParser()
    data2 = mergeParser.ObjectParser()
     
    # Accessing class attributes
    # and method through objectIndex
    data1.fun(object_file1)
    data2.fun(object_file2)

    #merging data
    #defenition 'MergeData' list
    MergeData=[]

    #call 'merge' function and store the results (new_timeStep,new_objectIndex,new_ParentIndex,
    #new_ParentTimeStep,new_objectLable,new_xCenter,new_yCenter,new_orientation,new_bacLength,marker1,marker2)
    # to MergeData list
    # note: MergeData list is a nested list
    MergeData=mergeProcessing.merge(data1,data2)

    #write results to csv file
    with open('OutputFile/merge.csv', 'w',newline='') as myfile:
        wr = csv.writer(myfile)
        #header          
        wr.writerow(['ImageNumber',color_num1,color_num2,'ObjectNumber',"AreaShape_MajorAxisLength","AreaShape_Orientation","Location_Center_X","Location_Center_Y",
                     "TrackObjects_Label_50","TrackObjects_ParentImageNumber_50","TrackObjects_ParentObjectNumber_50"])
        #rows
        for i in range(len(MergeData[0])):
            wr.writerow([MergeData[0][i],MergeData[9][i],MergeData[10][i],MergeData[1][i],MergeData[8][i],MergeData[7][i],MergeData[5][i],MergeData[6][i],MergeData[4][i],MergeData[3][i],
                         MergeData[2][i]])

    print('Two Strain CSV files were merged!')
    print('The merged file was written in the OutputFile folder!')
