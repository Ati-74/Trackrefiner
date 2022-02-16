import csv
import math
import operator

#creation ObjectParser Class
class ObjectParser():

   #definition of Class Attributes
   def __init__(self):
      self.timeStep = []
      self.objectIndex = []
      self.ParentIndex=[]
      self.ParentTimeStep=[]
      self.objectLable=[]
      self.xCenter=[]
      self.yCenter=[]
      self.orientation=[]   
      self.bacLength=[]
      
   #Parsing CellProfiler output 
   def fun(self,csvfile):
   
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
            # extracting field names through first row
            header = next(csvreader)
            # extracting each data row one by one
            for row in csvreader:
                rows.append(row)
    #Fetching data
    #row: bacteria features value
    #object_num: bacteria index in rows list
        for object_num,row in enumerate(rows):
            #finding value of important features for each bacteria
            #important features: ImageNumber,ObjectNumber,TrackObjects_ParentObjectNumber_50,TrackObjects_ParentImageNumber_50
            #important features: TrackObjects_Label_50,Location_Center_X,Location_Center_Y,AreaShape_Orientation,
            #important features: AreaShape_MajorAxisLength
            #important functions:
            # header.index(...): finding the index of feature in 'header' list
            # `attribute`.append: appending value to end of `attribute` list 
            self.timeStep.append(int(row[header.index("ImageNumber")]))
            self.objectIndex.append(int(row[header.index("ObjectNumber")]))
            self.ParentIndex.append(float(row[header.index("TrackObjects_ParentObjectNumber_50")]))
            self.ParentTimeStep.append(float(row[header.index("TrackObjects_ParentImageNumber_50")]))
            self.objectLable.append(int(row[header.index("TrackObjects_Label_50")]))
            self.xCenter.append(float(row[header.index("Location_Center_X")]))
            self.yCenter.append(float(row[header.index("Location_Center_Y")]))
            self.orientation.append(float(row[header.index("AreaShape_Orientation")]))
            self.bacLength.append(float(row[header.index("AreaShape_MajorAxisLength")]))
