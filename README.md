# INST0060 - Research Template 1 - Group 12

## About 

* This project compare 4 models' performance on a bank consuemr dataset. 
* The four models are:
- Logistic Regression 
- Random Forest
- KNN
- Fisher Linear Discriminant

## Installation

We provided a separate files with the required libraries to run the experiments:
<requirement.txt>

## Usage

To run the experiment use the following structure: 
python model_comparison.py <file.csv> <target_value> <model> <column_to_drop>

The experiment value options are: 
- Logistic_Regression 
- KNN
- Fisher 
- Random_Forest

EXAMPLE:
python model_comparison.py Churn_Modelling.csv Exited Random_Forest RowNumber,CustomerId,Surname 

### Content

Description, sub-modules organization...

### Requirements

