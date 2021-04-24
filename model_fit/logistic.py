import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd


from fomlads.data.external import standard_scaler
from fomlads.model.classification import split_train_test
from fomlads.evaluate.eval_classification import two_class_cf_matrix, roc , roc_auc, two_class_f1_score
from fomlads.plot.evaluations import plot_train_test_errors
from fomlads.evaluate.eval_classification import misclassification_error
from fomlads.model.classification import logistic_regression_fit
from fomlads.model.classification import logistic_regression_predict, logistic_regression_prediction_probs

def logistic_hyperparameters(train_inputs, train_targets,validation_inputs, validation_targets ):
    binary_col = [*train_inputs.loc[:,train_inputs.isin([0,1]).all()].columns]
    train_inputs = train_inputs.drop(binary_col, axis = 1) 
    train_inputs = train_inputs.to_numpy()  
    train_targets = train_targets.to_numpy()
    train_inputs = standard_scaler(train_inputs)

     #Test the parameters
    reg_params = np.linspace(0, 2)
    train_errors, test_errors = test_parameter_logistic(train_inputs, train_targets, validation_inputs, validation_targets , parameter_values= reg_params, parameter_name = 'Threshold')
    
    #Plot the test and train errors 
    plot_train_test_errors('Threshold', reg_params, train_errors, test_errors)

def fit_evaluate_logistic(train_inputs, train_targets, test_inputs, test_targets):
    """
    Takes train and test data as input 
    Fits a logistic regression 
    Return F1 score, confusion matrix and AUC ROC 
    """

    #drop the catagorical encodings 
    binary_col = [*train_inputs.loc[:,train_inputs.isin([0,1]).all()].columns]
    train_inputs = train_inputs.drop(binary_col, axis = 1)
    test_inputs = test_inputs.drop(binary_col, axis=1)

    #transform pd dataframe into numpy array
    train_inputs = train_inputs.to_numpy()
    train_targets = train_targets.to_numpy()
    test_inputs = test_inputs.to_numpy()
    test_targets= test_targets.to_numpy()

    train_inputs = standard_scaler(train_inputs)
    test_inputs = standard_scaler(test_inputs)

    #fit the model to the train data
    weights = logistic_regression_fit(train_inputs, train_targets, threshold = 1e-8)
    #Get the prediction for the test data
    y_pred = logistic_regression_predict(test_inputs, weights)
    #Metrics
    print("Metrics:")

    #CF Matrix
    print("Confusion Matrix")
    cf = two_class_cf_matrix(test_targets, y_pred)
    print(cf)

    #F1 Score
    print("F1_Score")
    f1 = two_class_f1_score(test_targets, y_pred)
    print(f1)


    plt.figure(figsize=(15,7))
    prob_vector = logistic_regression_prediction_probs(test_inputs, weights)
    #print(prob_vector)
    print(len(prob_vector))
    ROC = roc(prob_vector,test_targets, partitions=100)
    #print(ROC)
    plt.scatter(ROC[:,0],ROC[:,1],color='#0F9D58',s=30)
    plt.title('ROC Curve',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    fpr, tpr = ROC[:, 0], ROC[:, 1]
    plt.show()

    #AUC-ROC
    auc = roc_auc(prob_vector,test_targets, partitions=100)
    print('Area under curve={}'.format(auc))
    plt.show()


    plt.show()

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






