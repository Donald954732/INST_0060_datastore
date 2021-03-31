import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain


import numpy as np 

from fomlads.data.external import import_for_classification
from fomlads.data.external import normalisation
from fomlads.plot.evaluations import plot_roc
from fomlads.plot.evaluations import plot_train_test_errors
from fomlads.evaluate.eval_classification import confusion_matrix
from fomlads.evaluate.eval_classification import misclassification_error
from fomlads.evaluate.eval_logistic import test_parameter_logistic
from fomlads.evaluate.eval_classification import false_true_rates

from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict
from fomlads.model.classification import logistic_regression_prediction_probs

from fomlads.model.regression import construct_feature_mapping_approx
from fomlads.model.basis_functions import quadratic_feature_mapping


def main():

    #Read the file and clean the data
    churn_data = pd.read_csv('Churn_Modelling.csv')
    word_label = ['Geography', 'Gender']
    removing = []

    for column in word_label:
        if(churn_data[column].dtype == np.str or churn_data[column].dtype == np.object):
            for cat in churn_data[column].unique():
                churn_data[column+'_'+cat] = np.where(churn_data[column] == cat, 1, 0)
            removing.append(column)
    churn_data = churn_data.drop(removing, axis = 1)

    churn_data = churn_data.drop(['RowNumber', 'CustomerId', 'Surname'], axis =1)

    #Normalise the data for the logistic regression 
    need_normalization = [ 'CreditScore', 'Age', 'Tenure', 'Balance','NumOfProducts','EstimatedSalary']
    churn_data[need_normalization] = normalisation(churn_data[need_normalization])

    #FIRST TRY:
    # Using Basis Function - RBF
    centres = np.linspace(0,1,9)
    scale = 0.1
    #feature_mapping = construct_rbf_feature_mapping(centres, scale)
    #designmtx = quadratic_feature_mapping(inputs)
    inputs= churn_data[['CreditScore', 'Age', 'Tenure','Balance','NumOfProducts', 'EstimatedSalary']].to_numpy()
    targets = churn_data['Exited'].to_numpy()


    fig_ax = fit_and_plot_roc_confusion_matrix_logistic_regression(inputs, targets, colour = 'b')



    #TRAIN AND TEST DATA AND THRESHOLD 
    train_set, test_set = split_train_test(churn_data, test_ratio= 0.2)

        #Define inputs and outputs for both sets:
    train_inputs = train_set[['CreditScore', 'Age', 'Tenure','Balance','NumOfProducts', 'EstimatedSalary']].to_numpy()
    train_targets = train_set['Exited'].to_numpy()
    test_inputs = test_set[['CreditScore', 'Age', 'Tenure','Balance','NumOfProducts', 'EstimatedSalary']].to_numpy()
    test_targets = test_set['Exited'].to_numpy()

        #Need to change that or check it's a good idea to use that 
    reg_params = np.linspace(0, 2)

    test_parameter_logistic(train_inputs, train_targets, test_inputs, test_targets, parameter_values= reg_params)
    

    plt.show()


def fit_and_plot_roc_confusion_matrix_logistic_regression(
        inputs, targets, fig_ax=None, colour=None):
    """
    Takes input and target data for classification and fits shared covariance
    model. Then plots the ROC corresponding to the fit model.

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1
    """
    weights = logistic_regression_fit(inputs, targets)
    #
    thresholds = np.linspace(0,1.5,1000)
    

    false_positive_rates, true_positive_rates = false_true_rates(inputs, targets, weights, thresholds)

    fig1, ax1 = plot_roc(
        false_positive_rates, true_positive_rates, fig_ax=fig_ax, colour=colour)


    # and for the class prior we learnt from the model
    num_neg = np.sum(1-targets)
    num_pos = np.sum(targets)
    predicts = logistic_regression_predict(inputs, weights)
    fpr = np.sum((predicts == 1) & (targets == 0))/num_neg
    tpr = np.sum((predicts == 1) & (targets == 1))/num_pos
    ax1.plot([fpr], [tpr], 'rx', markersize=8, markeredgewidth=2)

    #Confusion matrix
    fig2, ax2 = confusion_matrix(targets, predicts)
    return (fig1, fig2), (ax1, ax2)


# For illustration only. Scikit-learn has train_test_split()
def split_train_test(data, test_ratio = 0.25):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == "__main__":
    
        main()
