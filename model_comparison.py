import pandas as pd
import numpy as np
from model_fit.logistic import fit_evaluate_logistic
from model_fit.RandomForest import rf_main
from model_fit.KNN_classifier import knn_main
from model_fit.fisher import fisher_main
from fomlads.data.preprocessing import pre_process



def main(ifname, target_col=None, experiment = None, column_to_drop = None):
     """
     TO CALL: 
python model_comparison.py Churn_Modelling.csv Exited Logistic_Regression RowNumber,CustomerId,Surname 
    Imports the data-set and generates exploratory plots
    ifname -- filename/path of data file.
    input_cols -- list of column names for the input data
    target_col -- column name of the target data
    classes -- list of the classes to plot
    experiment -- name of the model you want to test (Logistic_Regression, Random_Forest, Fisher, KNN)
    """
     
     dataframe = pd.read_csv(ifname)
  
     
     test_ratio = 0.2
     X_train, y_train, X_test, y_test = pre_process(dataframe, column_to_drop, target_col, test_ratio)

     if experiment == "Logistic_Regression":
         fit_evaluate_logistic(X_train, y_train, X_test, y_test)
     elif experiment == "Random_Forest":
         rf_main(X_train, y_train, X_test, y_test)
     elif experiment == "KNN":
         knn_main(X_train, y_train, X_test, y_test)
     elif experiment == "Fisher":
         fisher_main(X_train, y_train, X_test, y_test)



if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        main() # calls the main function with no arguments
    else:
        # assumes that the first argument is the input filename/path
        if len(sys.argv) == 2:
            main(ifname=sys.argv[1])
        else:
            target_col = sys.argv[2]
            if len(sys.argv) == 3:
                main(
                    ifname=sys.argv[1],
                    target_col=target_col)
            # assumes that the fourth argument is the list of input columns
            else:
                experiment = sys.argv[3]
                if len(sys.argv) == 4:
                    main(
                        ifname=sys.argv[1], 
                        target_col=target_col, experiment=experiment)
                else:
                    column_to_drop = sys.argv[4].split(',')
                    main(
                        ifname=sys.argv[1],  column_to_drop=column_to_drop,
                        target_col=target_col, experiment=experiment)