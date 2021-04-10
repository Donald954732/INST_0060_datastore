import pandas as pd
import numpy as np

import sys

'''
This module pre process that data my removing column out of scope and encoding word labe column in one hot vector
run:
python pre-processing.py dataset
'''

np.random.seed(123)

input_data = pd.read_csv(sys.argv[1])
input_data.head()

print("Checking Data")
print('\nNull Values \n{}'.format(input_data.isnull().sum()))

print('\nDuplicated values {}'.format(input_data.duplicated().sum()))

column_to_drop = list(input("Input the column to drop :").split(','))
print("Dropping column:", column_to_drop)

input_data = input_data.drop(column_to_drop, axis=1)

word_label = list(input("Input the word label columns ").split(','))
print("Category Label:", word_label)
removing = []
for column in word_label:
    if(input_data[column].dtype == np.str or input_data[column].dtype == np.object):
        for cat in input_data[column].unique():
            input_data[column+'_'+cat] = np.where(input_data[column] == cat, 1, 0)
        removing.append(column)
input_data = input_data.drop(removing, axis = 1)

print("Data after pre-processing")
print(input_data.head())
input_data.to_csv("processed.csv")
