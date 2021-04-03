import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fomlads.evaluate.eval_classification import misclassification_error
from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict
from fomlads.model.classification import logistic_regression_prediction_probs
from fomlads.plot.evaluations import plot_train_test_errors
from fomlads.model.classification import split_train_test
from fomlads.evaluate.eval_classification import cross_entropy_error

def test_parameter_logistic(train_inputs, train_targets, test_inputs, test_targets, parameter_values= np.linspace(0, 2), parameter_name = 'Threshold'):
    """ 
    train_inputs
    train_targets
    test_inputs
    train_inputs


    return
    ----- 
    plot of test/train error according to the parameter tested
    """
    
    train_errors = []
    test_errors = []

    for reg_param in parameter_values:
            ## Calculate wieghts on the training data 
        weights = logistic_regression_fit(train_inputs, train_targets)
        
            #get preduction for both data sets
        prediction_probs_train = logistic_regression_prediction_probs(train_inputs, weights)
        prediction_probs_test = logistic_regression_prediction_probs(test_inputs, weights)

        predicts_train = (prediction_probs_train > reg_param).astype(int)
        predicts_test = (prediction_probs_test > reg_param).astype(int)

            #Get the errors for both data sets
        train_error = misclassification_error(train_targets, predicts_train)
        test_error =  misclassification_error(test_targets, predicts_test)
        train_errors.append(train_error)
        test_errors.append(test_error)
    
    return train_errors, test_errors


def cross_validation(inputs, targets, weights, k=4):
    """
    Takes inputs and outputs and run a cross validation test on them.
    Fit the model and evaluate the test and train error 
    inputs - input data
    targets - target data
    weights - when the weights have been established 
    k - number of fold 

    Return
    -----------
    returns  fig of error for test and train of all the folds 
    """
     
    shuffled_indices= np.random.permutation(len(inputs))
    size = int(len(inputs)/k)


    train_errors = []
    test_errors =[]

    for i in np.arange(1, k): 
        #Take data from only one of the splits
        start = size*(i-1)
        stop = size*i
        indices = shuffled_indices[start:stop]
        inputs_1 = inputs.iloc[indices]
        targets_1 = targets.iloc[indices]
        ## Split dataframe into train and test 
        split_indices = np.random.permutation(len(inputs_1))
        test_set_size = int(len(inputs_1) * 0.25)
        test_indices = split_indices[:test_set_size]
        train_indices = split_indices[test_set_size:]
        
        train_inputs = inputs.iloc[train_indices]
        train_targets = targets.iloc[train_indices]

        test_inputs = inputs.iloc[test_indices]
        test_targets = targets.iloc[test_indices]
        
        train_predict_probs = logistic_regression_prediction_probs(train_inputs, weights)
        test_predict_probs = logistic_regression_prediction_probs(test_inputs, weights)

        train_error = cross_entropy_error(train_targets, train_predict_probs)
        test_error = cross_entropy_error(test_targets, test_predict_probs)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plot_train_test_errors("cross validation", np.arange(k), train_errors, test_errors)
    



