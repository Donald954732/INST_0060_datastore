import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fomlads.evaluate.eval_classification import misclassification_error
from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict
from fomlads.model.classification import logistic_regression_prediction_probs

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






