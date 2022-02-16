import csv

#creating ObjectParser Class
#object parser is a class that contains attributes and functions. By "object parser", I store important columns of CellProfiler output in lists defined below.
#class of "object parser" has two def; first one which is "def_init_(self) is for defining the attributes and second one (def pasrse) is method of class.
class ObjectParser():
   #these attributes are required to find lineage of bacteria, elongation rate calculation
   def __init__(self):
      self.time_steps = []
      self.object_index = []
      self.Parent_index=[]
      self.Parent_time_step=[]
      self.object_lable=[]
      self.length=[]
      #I defined all_bac_row_index list to have global index for all bacteria; 
      #global index in previous line means it is independent from object number and 
      #by calling that I can get access to the value of useful features of bacteria (those that are important for bacterial lineage,...)
      #I can't use self.object_index for this purpose because self.object_index always reset to 1 in each timestep
      #row number of all bacteria in CSV file
      self.all_bac_row_index=[]      
      
   #Parsing CellProfiler output 
   def Parse(self,csvfile):      
   #imagine a data matrix
   #header   feature1       feature2      feature3
   #row1    bac1Value1    bac1Value2    bac1Value3
   #row1    bac2Value1    bac2Value2    bac2Value3
   #row1    bac3Value1    bac3Value2    bac3Value3  
        # initializing the header and rows list
        header = []
        rows = []       
        # reading csv file
        with open(csvfile+'.csv', 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)        
            #storing headers of columns in the list named header
            header = next(csvreader)
            #storing row data in the list named row
            #Add row values of each bacteria to the end of rows list
            for row in csvreader:
                rows.append(row)
    #pupose of following "for": appending values to the defined attributes.
    #row: bacteria features value
    #counter: index(number) of row which starts from zero    
        for counter,row in enumerate(rows):
            #finding value of useful features for each bacteria
            #useful features: ImageNumber,ObjectNumber,TrackObjects_ParentObjectNumber_50,TrackObjects_ParentImageNumber_50
            #useful features: TrackObjects_Label_50,
            #useful features: AreaShape_MajorAxisLength            
            self.time_steps.append(int(row[header.index("ImageNumber")]))
            self.object_index.append(int(row[header.index("ObjectNumber")]))
            self.Parent_index.append(float(row[header.index("TrackObjects_ParentObjectNumber_50")]))
            self.Parent_time_step.append(float(row[header.index("TrackObjects_ParentImageNumber_50")]))
            self.object_lable.append(int(row[header.index("TrackObjects_Label_50")]))
            self.length.append(float(row[header.index("AreaShape_MajorAxisLength")]))
            # appending row number of bacteria in csv file in list named "all_bac_row_index"
            self.all_bac_row_index.append(counter)
    
