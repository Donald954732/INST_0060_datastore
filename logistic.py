import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain


import numpy as np 

from fomlads.data.external import import_for_classification
from fomlads.data.external import normalisation
from fomlads.plot.evaluations import plot_roc
from fomlads.plot.evaluations import plot_train_test_errors
from fomlads.evaluate.eval_classification import confusion_matrix
from fomlads.evaluate.eval_logistic import test_parameter_logistic
from fomlads.evaluate.eval_classification import false_true_rates
from fomlads.evaluate.eval_classification import f1_score

from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict
from fomlads.model.classification import split_train_test


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

    #Define inputs and targets in the dataframe
    inputs= churn_data[need_normalization].to_numpy()
    targets = churn_data['Exited'].to_numpy()

    fit_evaluate_logistic(inputs, targets, churn_data)

def fit_evaluate_logistic(inputs, targets, data):

    #fit the data
    weights = logistic_regression_fit(inputs, targets, threshold = 1e-8)
    predicts = logistic_regression_predict(inputs, weights)
 
    # Plot the corresponding ROC 
    thresholds = np.linspace(0,1.5,100)
    false_positive_rates, true_positive_rates = false_true_rates(inputs, targets, weights, thresholds)
    fig1, ax1 = plot_roc(
        false_positive_rates, true_positive_rates)
       # and for the class prior we learnt from the model
    num_neg = np.sum(1-targets)
    num_pos = np.sum(targets)
    fpr = np.sum((predicts == 1) & (targets == 0))/num_neg
    tpr = np.sum((predicts == 1) & (targets == 1))/num_pos
    ax1.plot([fpr], [tpr], 'rx', markersize=8, markeredgewidth=2, color = 'b')

    # Plot the confusion matrix
    fig2, ax2 = confusion_matrix(targets, predicts)

    #Split the dataset into test and train sets
    train_set, test_set = split_train_test(data, test_ratio= 0.2)

        #Define inputs and outputs for both sets:
    train_inputs = train_set[['CreditScore', 'Age', 'Tenure','Balance','NumOfProducts', 'EstimatedSalary']].to_numpy()
    train_targets = train_set['Exited'].to_numpy()
    test_inputs = test_set[['CreditScore', 'Age', 'Tenure','Balance','NumOfProducts', 'EstimatedSalary']].to_numpy()
    test_targets = test_set['Exited'].to_numpy()

    #Get the test and train errors 
    reg_params = np.linspace(0, 0.75)
    train_errors, test_errors = test_parameter_logistic(train_inputs, train_targets, test_inputs, test_targets, parameter_values= reg_params)
    
    #Plot the test and train errors 
    plot_train_test_errors('Threshold', reg_params, train_errors, test_errors)

    plt.show()


    f1 = f1_score(targets, predicts)
    print(f1)


if __name__ == "__main__":
    
        main()
