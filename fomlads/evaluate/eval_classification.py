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

def confusion_matrix(targets, y_prediction):
    """
    Creating and plotting a confusion matrix 

    targets - actual y as a pd dataframe
    y_prediciton - prediction as a pd dataframe

    returns
    -------
    fig, ax of the confusion matrix
    """
    K = len(np.unique(targets))
    result = np.zeros((K,K))
    
    
    for i in range(len(targets)):
        result[targets[i]][y_prediction[i]] += 1


    def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix', cmap=plt.cm.gist_heat):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)


        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #    plt.text(int(j), int(i), format(cm[i, j], fmt),
        #            horizontalalignment="center",
        #            color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout() 
        return fig, ax


    return plot_confusion_matrix(result, np.unique(targets))


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

def score(targets, predicts):
    K = len(np.unique(targets))
    result = np.zeros((K,K))
    for i in range(len(targets)):
        result[targets[i]][predicts[i]] += 1



    correct_prediction = result[0][0] + result[1][1]
    total_prediction = np.size(predicts)
    true_positives = result[0][0]
    predicted_postives = result[0][0] + result[0][1]
    actual_positves = result[0][0] + result[1][0]
    precision = true_positives/predicted_postives
    recall = true_positives/ actual_positves
    f1=  2*(precision* recall)/(precision+ recall)
    accuracy = correct_prediction/total_prediction
    return f1, precision, recall, accuracy


