# TrackRefiner Algorithm’s Workflow

The flowcharts illustrating the workflow of the TrackRefiner algorithm are presented below. </br></br>
Standard flowchart conventions are employed, as follows:
- Oval (Terminator): Denotes the initiation or termination of the workflow.
- Rectangle (Process): Represents a specific step, action, or operation.
- Diamond (Decision): Indicates a decision point, typically associated with “Yes” or “No” outcomes.
- Parallelogram (Input/Output): Depicts data entry or output operations.
- Arrow: Connects elements and specifies the direction of process flow.

For enhanced clarity, color coding is applied so that <b>blue indicates correction processes</b> and <b>yellow denotes identification steps</b>.

## Lineage Based Error Identification

<img src="01.Lineage%20Based%20Error%20Identification.jpg" alt="Lineage Based Error Identification" width="100%">

## ML Model Training
<p align="justify">
To improve clarity, this section has been separated into two flowcharts. In the flowchart for the ML section called Classifier Training 1, there is a process labeled Train Classifier. The second flowchart, titled Classifier Training 2, presents this training process in greater detail. Since the procedure is
identical for all three classifiers and involves many steps, we created a separate flowchart to avoid
repetition and make both flowcharts easier to follow.
</p>

### Classifier Training 1
<img src="02.1.Classifier training.jpg" alt="02.1.Classifier training.jpg" width="100%">

### Classifier Training 2
<img src="02.2.Classifier training.jpg" alt="02.2.Classifier training.jpg" width="100%">


## Feature-Based Identification and Correction

<p align="justify">
This section presents five sequential flowcharts, each representing a specific type of error in the order they are addressed during implementation.
In steps <i>Swapped track links and missing division links</i> and <i>Untimely root and leaf</i>, there is a box labeled Evaluate the feasibility of making new links based on biological limits, data-driven thresholds, and the likelihoods predicted by the models. This box is further detailed in a separate flowchart, corresponding to <i>Applying Classifiers</i>.
</p>

### Feature-Based Identification and Correction for over-assigned division links
<img src="03.1.over-assigned division links.jpg" alt="03.1.over-assigned division links.jpg" width="100%">

### Feature-Based Identification and Correction for Redundant division links
<img src="03.2.Redundant division links.jpg" alt="03.2.Redundant division links.jpg" width="100%">

### Feature-Based Identification and Correction Flow chart for Swapped track links and Missing division links
<img src="03.3.Swapped track links and missing division links.jpg" alt="03.3.Swapped track links and missing division links.jpg" width="100%">

### Feature-Based Identification and Correction for Untimely root and leaf
<img src="03.4.untimely root and leaf.jpg" alt="03.4.untimely root and leaf.jpg" width="100%">


### Applying Classifiers
<img src="03.5.Applying Classifiers.jpg" alt="03.5.Applying Classifiers.jpg" width="100%">
