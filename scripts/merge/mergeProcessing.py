import csv
import math
import operator


def merge(data1,data2):

    #create dictionary
    # dictionary structure:  key:value
    # key: value (index in Dic_index and lable in Dic_lable1) in the previous list
    # value: value in the merge list
    # dictionaries of first file
    Dic_index1={0:0}
    Dic_lable1={}
    # dictionaries of second file
    Dic_index2={0:0}
    Dic_lable2={}

    # max index value in each bacteria type
    max_index_value1=0
    max_index_value2=0
    # max lable value in each bacteria type
    max_lable_value1=0
    max_lable_value2=0

    # getting unique time steps values
    unique_timeStep = set(data1.timeStep)

    # lists of final values of bacteria
    #note: we should update the value of: objectIndex,ParentIndex,and  objectLable of bacteria
    new_timeStep = []
    new_objectIndex = []
    new_ParentIndex=[]
    new_ParentTimeStep=[]
    new_objectLable=[]
    new_xCenter=[]
    new_yCenter=[]
    new_orientation=[]
    new_bacLength=[]
    #marker list elements value: 0 or 1
    #if bacteria are in type 1: the related element value in `marker1` list is 1 
    #and the related element value in `marker1` list is 0
    marker1=[]
    marker2=[]
    
    for time in unique_timeStep:
        #obj: bacteria index value (related to 'ObjectNumber' column in CellProfiler)
        #i: index of bacteria ObjectNumber value in 'objectIndex' list    
        for i,obj in enumerate(data1.objectIndex):
            if data1.timeStep[i]==time:
            
              #new ObjectNumber of bacteria 
              if obj in Dic_index1:
                  new_objectIndex.append(Dic_index1[obj])
              else:
                  #calculating new ObjectNumber value of bacteria 
                  max_value=max(max_index_value1,max_index_value2)+1
                  #append 'max_value' to 'new_objectIndex' list
                  new_objectIndex.append(max_value)
                  #update the max_index_value1
                  #max_index_value1=latest index of bacteria type 1
                  max_index_value1=max_value
                  #add new ObjectNumber  to 'Dic_index1' dictionary
                  #it's used to find new ParentObjectNumber of daughter bacteria
                  Dic_index1[obj]=max_value 
              
              #appending new parent object number
              new_ParentIndex.append(Dic_index1[data1.ParentIndex[i]])
                  
              #finding new bacteria lable
              if data1.objectLable[i] in Dic_lable1:
                  new_objectLable.append(Dic_lable1[data1.objectLable[i]])
              else:
                  #calculating new lable value of bacteria 
                  max_value=max(max_lable_value1,max_lable_value2)+1
                  #append 'max_value' to 'new_objectLable' list
                  new_objectLable.append(max_value)
                  #update the max_lable_value1
                  #max_lable_value1=latest lable of bacteria type 1                  
                  max_lable_value1=max_value
                  #add new ObjectNumber  to 'Dic_lable1' dictionary
                  #it's used for lineage bacteria              
                  Dic_lable1[data1.objectLable[i]]=max_value
                  
              #copy the value of 'ImageNumber,TrackObjects_ParentImageNumber_50
              # Location_Center_X,Location_Center_Y,AreaShape_Orientation,AreaShape_MajorAxisLength'
              #from previous lists to new lists
              new_timeStep.append(data1.timeStep[i])
              new_ParentTimeStep.append(data1.ParentTimeStep[i])
              new_xCenter.append(data1.xCenter[i])
              new_yCenter.append(data1.yCenter[i])
              new_orientation.append(data1.orientation[i])
              new_bacLength.append(data1.bacLength[i])
              #appending one or zero to markers list according to bacteria type
              marker1.append(1)
              marker2.append(0)

        #same as line 107 to 154
        for i,obj in enumerate(data2.objectIndex):
            if data2.timeStep[i]==time:
              if obj in Dic_index2:
                  new_objectIndex.append(Dic_index2[obj])
              else:
                  max_value=max(max_index_value1,max_index_value2)+1
                  new_objectIndex.append(max_value)
                  max_index_value2=max_value
                  Dic_index2[obj]=max_value
                  
              #parent object number             
              new_ParentIndex.append(Dic_index2[data2.ParentIndex[i]])
                  
              if data2.objectLable[i] in Dic_lable2:
                  new_objectLable.append(Dic_lable2[data2.objectLable[i]])
              else:
                  max_value=max(max_lable_value1,max_lable_value2)+1
                  new_objectLable.append(max_value)
                  max_lable_value2=max_value
                  Dic_lable2[data2.objectLable[i]]=max_value
                  
              new_timeStep.append(data2.timeStep[i])
              new_ParentTimeStep.append(data2.ParentTimeStep[i])
              new_xCenter.append(data2.xCenter[i])
              new_yCenter.append(data2.yCenter[i])
              new_orientation.append(data2.orientation[i])
              new_bacLength.append(data2.bacLength[i])
              marker1.append(0)
              marker2.append(1)              

    return [new_timeStep,new_objectIndex,new_ParentIndex,new_ParentTimeStep,new_objectLable,new_xCenter,new_yCenter,new_orientation,new_bacLength,marker1,marker2]
