# INST0060 - Research Template 1 - Group 12

## About 

* This project compare 4 models' performance on a bank consumers dataset. 
* The four models are:
- Logistic Regression 
- Random Forest
- KNN
- Fisher Linear Discriminant

## Installation

We provided a separate files with the required libraries to run the experiments:
<requirement.txt>
The steps to follow to create the relevant environment are in the Requirements section below. 

## Usage

To run the experiment use the following syntax on your machine's terminal: 
python model_comparison.py <file.csv> <target_value> <model> <column_to_drop>

The experiment value options are: 
- Logistic_Regression ~ ( runtime 1 to 2 min)
- KNN ~ ( runtime 1 min)
- Fisher ~ ( runtime 1 to 2 min)
- Random_Forest ~ ( runtime 1 min)

EXAMPLE:
python model_comparison.py Churn_Modelling.csv Exited Random_Forest RowNumber,CustomerId,Surname 

### Content

The project is structure with: 
- model_comparison.py ~ main method which calls methods to fit and evaluate each models
- model_fit folder ~ contains one file for each model. Each file has a main method called in model_comparison.py and the methods used to fit the models 
- fomlads.evaluate.eval_classification.py ~ contains methods to evaluate the models 
- fomlads.data.preprocessing.py ~ contains method used to pre-process the raw data file
- fomlads.data.external.py ~ contains the methods used to standardise and normalise the data
### Requirements
The requirements to run these experiments are contained in the <requirements.txt> file. 
This file should be used to create an environment following the steps below: 
1. conda create -n <name_of_your_environment> python=3.7
2. conda activate <name_of_your_environment>
3. python -m pip install -r requirements.txt
* the requirements.txt file should be in the same directory your currently working on 