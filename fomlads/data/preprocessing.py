import pandas as pd
import numpy as np
from fomlads.model.classification import split_train_test
import sys


def pre_process(input_data, column_to_drop, target_column, test_ratio, strategy = 'oversample'):
    """
    Takes the dataset as well as the column which will not be used and the target and the test ratio
    Checks for null values and duplicated values
    Performs one hot vectors encoding 
    Apply the relevant strategy to the data to over or under sample and deal with the imbalanced classes. 

    Returns the data split into train and test inputs and targets. 

    """

    np.random.seed(123)

    print("Checking Data")
    print('\nNull Values \n{}'.format(input_data.isnull().sum()))

    print('\nDuplicated values {}'.format(input_data.duplicated().sum()))

   
    print("Dropping column:", column_to_drop)

    input_data = input_data.drop(column_to_drop, axis=1)
    
    #One-Hot vector encoding for categorical data
    removing = []
    for column in input_data:
        if(input_data[column].dtype == np.str or input_data[column].dtype == np.object):
            for cat in input_data[column].unique():
                input_data[column+'_'+cat] = np.where(input_data[column] == cat, 1, 0)
            removing.append(column)
    input_data = input_data.drop(removing, axis = 1)

    print("Data after pre-processing")
    print(input_data.head())



    #Split Data
    
    train_set, test_set = split_train_test(input_data, test_ratio=test_ratio)
    X_test = test_set.drop([target_column], axis=1)
    y_test = test_set[target_column]

    count_class_0, count_class_1 = train_set.Exited.value_counts()

    df_class_0 = train_set[train_set[target_column] == 0]
    df_class_1 = train_set[train_set[target_column] == 1]

    print("classs count in train")
    print("0: ",count_class_0)
    print("1: ",count_class_1)

    #Strategies to deal with imbalanced classes
    if strategy == "undersample":
        df_class_0_under = df_class_0.sample(count_class_1)
        df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)
        print('Random under-sampling:')
        print(df_train_under.Exited.value_counts())
        X_train = df_train_under.drop([target_column], axis=1)
        y_train = df_train_under[target_column]
    elif strategy == "oversample":
        df_class_1_over = df_class_1.sample(count_class_0, replace=True)
        df_train_over = pd.concat([df_class_0, df_class_1_over], axis=0)
        print('Random over-sampling:')
        print(df_train_over.Exited.value_counts())
        X_train = df_train_over.drop([target_column], axis=1)
        y_train = df_train_over[target_column]
    else:
        print("no strategy")
        X_train = train_set.drop([target_column], axis=1)
        y_train = train_set[target_column]
    
    return X_train, y_train, X_test, y_test
