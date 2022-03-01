import math
import statistics
from collections import Counter
import numpy as np
from sklearn.linear_model import LinearRegression


#tracking bacteria,finding lineage of bacteria,calculating Elongation rate and Proximity
#ElongationProximity Arguments:
#           time_steps
#           all_bac_row_index
#           object_index
#           object_lable
#           length
#           Parent_index
#           Parent_time_step
#           x_location
#           y_location
#How it works:
#elongation rate: tracking each bacterium and calculate its elongation rate based on formuala: (ln(l(last))-ln(l(first))/age
#some odd types in calculating elongation rate: 1) life history =1 ---> I store elongation rate as NaN /// 2) some times CellProfiler reports the length
#of bacteria as zero, so in this case I store elongation rate as zero ///
#proximity: I find the minimum distance of each of these same bacteria in their time step and report its mean as 
# the average proximity.
#Furthermore:
#I find the first time step that the bacterium is born and the last time step that the bacterium exists,  
#also the length of bacteria in first time step of its life history and also the length of bacteria 
#in the last time step of its life history.

def ElongationProximity(growthrateMethod,interval_time,time_steps,all_bac_row_index,object_index,object_lable,length,Parent_index,Parent_time_step,marker2,x_location,y_location):

    #in the dictionary (below) I store:
    #"same" (nested list): I store index of each bacterium (for the duration of experiment) in one individual list.
    #"average_proximity"
    #in the output file, I report this information:
    #first_time_step
    #last_time_step 
    #final_lable
    #first_length
    #last_length
    #elongation_rate
    calculation_dict={'same':[],'average_proximity':[], 'first_time_step':[], 'last_time_step':[],
                        'final_lable':[], 'first_length':[], 'last_length':[],'elongation_rate':[] }
    
    #Checked is not nested list but it's like "same"; by "chacked" same bacteria will be considered just once!
    checked=[]

    #Difference of bacterial length during its life history.
    delta_length=0

    # getting unique time steps values
    unique_time_steps = list(set(time_steps))

    for i in all_bac_row_index:
       #checking the bacteria has been checked or not
       #This helps to consider the similar bacteria (before division) only once
       #e.g: bacteria 1 is present in 3 time steps: 1, 2, and 3. 
       #By tracking Bacteria 1, we no longer need to look for bacteria similar to Bacteria 1 
       #during the checking bacteria of time Step 2 and 3.
       #also, here I check the type of bacteria
       #finally, I calculate everything on bacteria type 1
       if i not in checked and marker2[i]==0:
           #creating of an internal list of bacteria i in same list
           #I created it to store index of all the same of bacteria to bacteria i in this internal list.
           calculation_dict['same'].append([i])

           #parent object index and parent image number of daughter bacteria 
           ParentIndex=object_index[i]
           ParentTimeStep=time_steps[i]
           
           #this loop continues till it couldn't find same bacteria; that means when it finds bacterial daughter it will terminate.
           for time in [obj_time_step for obj_time_step in unique_time_steps if obj_time_step>time_steps[i]]:
              #finding bacteria at next time step that are related to the bacteria i (founded bacteria can have two destination: same as i or its daughter) 
              lineage_obj_in_next_time_step=[lineage_obj for lineage_obj in all_bac_row_index if time_steps[lineage_obj]==time and Parent_index[lineage_obj]==ParentIndex and
               Parent_time_step[lineage_obj]==ParentTimeStep]
               
               
              #checking relativity type (daughter or same)
              if len(lineage_obj_in_next_time_step)>1:  #daughter
                 #termination of internal loop
                 #because: Because at this time step the bacterium is dead or has cell division. 
                 #Therefore, we no longer need to look for bacteria similar to Bacteria in the next time steps. 
                 #(Bacterial life is over)
                 break
              elif len(lineage_obj_in_next_time_step)==0:
                 break
              elif len(lineage_obj_in_next_time_step)==1: #same
                    #append the index of same bacteria to internal list of bacteria i of same list
                    #Always the last element of the same list refers to the current bacteria under investigation.
                    #last element index: len(calculation_dict['same'])-1
                    calculation_dict['same'][len(calculation_dict['same'])-1].extend(lineage_obj_in_next_time_step)
                    #append the index of same bacteria to checked list
                    #it helps us to have an optimized loop (reducing time complexity of "for" loop)
                    checked.extend(lineage_obj_in_next_time_step)
                    #update parent index number and parent object number
                    ParentIndex=object_index[lineage_obj_in_next_time_step[0]]
                    ParentTimeStep=time_steps[lineage_obj_in_next_time_step[0]]                    

           #length of same bacteria 
           length_obj=[length[j] for j in calculation_dict['same'][len(calculation_dict['same'])-1]]
           timestep_obj=[time_steps[j] for j in calculation_dict['same'][len(calculation_dict['same'])-1]]
           #convert to min
           timestep_obj = [t_obj * interval_time for t_obj in timestep_obj]
            
            #calculation of average closeness
            #I find the minimum distance of each of these same bacteria in their time step and report its mean as 
            # the average of proximity.
            #Arguments:
            #time_steps:
            #calculation_dict['same'][len(calculation_dict['same'])-1]: same bacteria to the current bacteria under investigation.
            #all_bac_row_index
            #marker2: I used that to find bacteri type 2
            #x_location
            #y_location
           avg_distance=closeness(time_steps,calculation_dict['same'][len(calculation_dict['same'])-1],all_bac_row_index,marker2,x_location,y_location)

           #calculation of elongation rate
           #length_obj[len(length_obj)-1]=division length
           #length_obj[0] =  length of bacteria when they are born
           #this condition checks the life history of bacteria
           #If the bacterium exists only one time step: NaN will be reported.
           if(len(length_obj)>1):
                if (growthrateMethod=="linearRegression"):
                    #linear regression
                    #convert to array
                    length_obj_regr=np.array(length_obj).reshape(-1, 1)
                    timestep_obj_regr=np.array(timestep_obj).reshape(-1, 1)
                    linear_regressor = LinearRegression()  # create object for the class
                    linear_regressor.fit(timestep_obj_regr,length_obj_regr)  # perform linear regression
                    calculation_dict['elongation_rate'].append(round(linear_regressor.coef_[0][0],3))
                    
                elif(growthrateMethod=="average"):
                    average_length=(math.log(length_obj[-1])-math.log(length_obj[0]))/(len(timestep_obj)*interval_time)
                    calculation_dict['elongation_rate'].append(round(average_length,3))
                
           else:
                calculation_dict['elongation_rate'].append("NaN") #shows: bacterium is present for only one timestep.
                
                
           #append average_proximity and more information (first_time_step,last_time_step,final_lable,first_length,last_length) to results lists   
           calculation_dict['average_proximity'].append(avg_distance)
           calculation_dict['first_time_step'].append(time_steps[i])
           calculation_dict['last_time_step'].append(time_steps[calculation_dict['same'][len(calculation_dict['same'])-1][len(calculation_dict['same'][len(calculation_dict['same'])-1])-1]])
           calculation_dict['final_lable'].append(object_lable[i])
           calculation_dict['first_length'].append(length_obj[0])
           calculation_dict['last_length'].append(length_obj[len(length_obj)-1])

    return calculation_dict
          

#calculation of average closeness
#I find the minimum distance of each of these same bacteria in their time step and report its mean as 
# the average of proximity.
#Arguments:
#time_steps:
#same_bac_index: same bacteria to the current bacteria under investigation.
#all_bac_row_index
#marker2: I used that to find bacteri type 2
#x_location
#y_location
def closeness(time_steps,same_bac_index,all_bac_row_index,marker2,x_location,y_location):

    #I save the distance of each bacterium type 1 from bacteria type 2 (in each time step) in the list below.
    distance_temp=[]
    #sum of min distance of bacteria type 1 to bacteria type 2 in each time step
    sum_distance=0
  
    #for each bacterium of type 1: finding the index of bacteria of type 2 (in each time step), note: this part is not dependent to time step but dependent to finding similar bacteria
    for object_same_bac_index in same_bac_index:
        same_time_steps_bac_index2=[counter for counter in all_bac_row_index if time_steps[counter]==time_steps[object_same_bac_index] and marker2[counter]==1 ]

        #calculating distance of bacteria to each bacterium of type 2 (in each time step)
        for object_bac_index2 in same_time_steps_bac_index2:
            #calculation of distance
            distance=math.sqrt(((x_location[object_same_bac_index]-x_location[object_bac_index2])**2)+
                    ((y_location[object_same_bac_index]-y_location[object_bac_index2])**2))
            #append to distance list
            distance_temp.append(distance)

        #sum of min distance 
        sum_distance=sum_distance+min(distance_temp)

        #clear distance_temp list
        distance_temp=[]
        
    average_proximity=sum_distance/len(same_bac_index)
        
    return(average_proximity)
