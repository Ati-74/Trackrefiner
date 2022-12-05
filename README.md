# CellProfilerAnalysis
Codes used to analyze CellProfiler output


## Converting CellProfiler data similar to CellModeller:
<div align="justify">
The process can be done by GUI. GUI Input file is CellProfiler output including columns:<br/> 
1. ImageNumber<br/>
2. ObjectNumber<br/>
3. AreaShape_MajorAxisLength<br/>
4. AreaShape_MinorAxisLength<br/>
5. AreaShape_Orientation<br/>
6. Intensity_MeanIntensity_rfp/gfp/yfp_channel (optional)<br/>
7. Location_Center_X (or AreaShape_Center_X), Location_Center_Y (or AreaShape_Center_Y)<br/>
8. TrackObjects_Label_50<br/>
9. TrackObjects_ParentImageNumber_50<br/>
10. TrackObjects_ParentObjectNumber_50<br/>
<br/><br/>
After choosing output file directory(same as input or wherever you select), and setting interval time based on your experiments and growth rate methods (average or linear regression) the process will be done by pressing start button. GUI final messages will guide you where the output is and whether the process is done. Final output is stored in both csv files and pickles for each time-step.
</div>
<br/>
<p align="center">
  <img src="doc/gui.png">
</p>
