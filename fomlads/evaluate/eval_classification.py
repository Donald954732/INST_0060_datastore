import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from fomlads.model.classification import logistic_regression_prediction_probs


def misclassification_error(targets, predicts):
    """
    Evaluate how closely predicted values (predict_probs) match the true
    values (targets) in a cross-entropy sense.

    Parameters
    ----------
    targets - the true targets a 1d array of 1s and 0s respectively
        corresponding to class 1 and 0
    predicts - the predictions, a 1d array  of 1s and 0s respectively
        predicting targets of class 1 and 0 

    Returns
    -------
    error - The minimum-misclassification error between true and predicted target
    """
    # flatten both arrays and ensure they are array objects
    targets = np.array(targets).flatten()
    predicts = np.array(predicts).flatten()
    N = targets.shape
    error = 1 - np.sum(targets == predicts)/N
    return error

def expected_loss(targets, predicts, lossmtx):
    """
    Evaluate how closely predicted values (predict_probs) match the true
    values (targets) in a cross-entropy sense.

    Parameters
    ----------
    targets - the true targets a 1d array of 1s and 0s respectively
        corresponding to class 1 and 0
    predicts - the predictions, a 1d array  of 1s and 0s respectively
        predicting targets of class 1 and 0 
    lossmtx - a matrix (2x2) of loss values for misclassification

    Returns
    -------
    error - An estimate of the expected loss between true and predicted target
    """
    # flatten both arrays and ensure they are array objects
    targets = np.array(targets).flatten()
    predicts = np.array(predicts).flatten()
    class0 = (targets == 0)
    class1 = np.invert(class0)
    predicts0 = (predicts == 0)
    predicts1 = np.invert(predicts0)
    class0loss = lossmtx[0,0]*np.sum(class0 & predicts0) \
        + lossmtx[0,1]*np.sum(class1 & predicts1)
    class1loss = lossmtx[1,0]*np.sum(class0 & predicts0) \
        + lossmtx[1,1]*np.sum(class1 & predicts1)
    N = targets.shape
    error = (class0loss + class1loss)/N
    return error


def cross_entropy_error(targets, predict_probs):
    """
    Evaluate how closely predicted values (predict_probs) match the true
    values (targets) in a cross-entropy sense.

    Parameters
    ----------
    targets - the true targets a 1d array of 1s and 0s respectively
        corresponding to class 1 and 0
    predict_probs - the prediction probabilities, a 1d array of probabilities 
        each predicting the probability of class 1 for the corresponding target

    Returns
    -------
    error - The cross-entropy error between true and predicted target
    """
    # flatted
    targets = np.array(targets).flatten()
    predict_probs = np.array(predict_probs).flatten()
    N = targets.shape
    error = - np.sum(
        targets*np.log(predict_probs) + (1-targets)*np.log(1-predict_probs))/N
    return error

def false_true_rates(inputs, targets,  weights, thresholds):
    """ 
    takes inputs and weights and get the prediction to then evaluate the number of false positive and true positive rates 

    return
    ----------
    false_positive_rates, true_positive_rates

    """
    N = targets.size
    num_neg = np.sum(1-targets)
    num_pos = np.sum(targets)

    #Plot the ROC
    false_positive_rates = np.empty(thresholds.size)
    true_positive_rates = np.empty(thresholds.size)
    for i, threshold in enumerate(thresholds):
        prediction_probs = logistic_regression_prediction_probs(inputs, weights)
        predicts = (prediction_probs > threshold).astype(int)
        num_false_positives = np.sum((predicts == 1) & (targets == 0))
        num_true_positives = np.sum((predicts == 1) & (targets == 1))
        false_positive_rates[i] = np.sum(num_false_positives)/num_neg
        true_positive_rates[i] = np.sum(num_true_positives)/num_pos
    return false_positive_rates, true_positive_rates



def two_class_cf_matrix(targets, predicts):
    '''
    return the confusion matrix of a 2 class prediction given target and prediction of models
    '''
    true_positives = np.sum((predicts == 1) & (targets == 1))
    true_negative = np.sum((predicts == 0) & (targets == 0))
    false_negative = np.sum((predicts == 0) & (targets == 1))
    false_positive = np.sum((predicts == 1) & (targets == 0))
    return np.array([[true_negative, false_positive],[false_negative, true_positives]])


def true_false_positive(threshold_vector, y_test):
    '''
    calculate the true positive and false positive rate given the threshold vector and probability vector of a classificaiton model
    '''

    true_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 1)
    true_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 0)
    false_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 0)
    false_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr


def roc(probabilities, y_test, partitions=100):
    '''
    generate the roc curve coordinate given probabilities and the varification data set
    '''
    roc = np.array([])
    for i in range(partitions + 1):
        #print(i)
        
        threshold_vector = np.greater_equal(probabilities, i / partitions)
        tpr, fpr = true_false_positive(threshold_vector, y_test)
        roc = np.append(roc, [fpr, tpr])
        
    return roc.reshape(-1, 2)

def roc_plot(prob_vector,y_test, partitions=100):
    """
    potting roc curve
    """
    plt.figure(figsize=(15,7))
    #print(prob_vector)
    print(len(prob_vector))
    ROC = roc(prob_vector,y_test, partitions=100)
    #print(ROC)
    plt.scatter(ROC[:,0],ROC[:,1],color='#0F9D58',s=30)
    plt.title('ROC Curve',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    fpr, tpr = ROC[:, 0], ROC[:, 1]
    integrated_roc = 0
    for k in range(partitions):
            integrated_roc = integrated_roc + (fpr[k]- fpr[k + 1]) * tpr[k]
    print("Area Under Curve of ROC: ", integrated_roc)
    plt.show()

def roc_auc(prob_vector,y_test, partitions=100):
    """
    calculate auc of ROC curve
    """
    ROC = roc(prob_vector,y_test, partitions=100)
    fpr_array = []
    tpr_array = []
    for i in range(len(ROC)-1):
        point1 = ROC[i]
        point2 = ROC[i+1]
        tpr_array.append([point1[0], point2[0]])
        fpr_array.append([point1[1], point2[1]])
    return sum(np.trapz(tpr_array,fpr_array))+1

def two_class_f1_score(targets, predicts):
    '''
    return the confusion matrix of a 2 class prediction given target and prediction of models
    '''
    true_positives = np.sum((predicts == 1) & (targets == 1))
    true_negative = np.sum((predicts == 0) & (targets == 0))
    false_negative = np.sum((predicts == 0) & (targets == 1))
    false_positive = np.sum((predicts == 1) & (targets == 0))
    Recall = true_positives / (true_positives+false_negative)
    Precision = true_positives / (true_positives+false_positive)
    return 2* (Precision*Recall)/(Precision + Recall)