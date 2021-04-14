from fomlads.data.external import import_for_classification
from logistic import fit_evaluate_logistic
from RandomForest import rf_main
from preprocessing import pre_process


def main(ifname, input_cols=None, target_col=None, classes=None, experiment = None ):
     """
    Imports the data-set and generates exploratory plots

    parameters
    ----------
    ifname -- filename/path of data file.
    input_cols -- list of column names for the input data
    target_col -- column name of the target data
    classes -- list of the classes to plot
    """

    inputs, targets, field_names, classes = import_for_classification(
        ifname, input_cols=input_cols, 
        target_col=target_col, classes=classes)

    X_train, y_train, X_test, y_test = pre_process(input_data, column_to_drop, word_label, target_column, test_ratio, strategy)

    if experiment = "Logistic_Regression":
        fit_evaluate_logistic(X_train, y_train, X_test, y_test)
    elif experiment = "Random_forest":
        rf_main(X_train, y_train, X_test, y_test, weight_balance)
    #elif experiment = "Fisher":
    #elif experiment = "KNN":



if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        main() # calls the main function with no arguments
    else:
        # assumes that the first argument is the input filename/path
        if len(sys.argv) == 2:
            main(ifname=sys.argv[1])
        else:
            # assumes that the second argument is a comma separated list of 
            # the classes to plot
            classes = sys.argv[2].split(',')
            if len(sys.argv) == 3:
                main(ifname=sys.argv[1], classes=classes)
            else:
                # assumes that the third argument is the list of target column
                target_col = sys.argv[3]
                if len(sys.argv) == 4:
                    main(
                        ifname=sys.argv[1], classes=classes,
                        target_col=target_col)
                # assumes that the fourth argument is the list of input columns
                else:
                    input_cols = sys.argv[4].split(',')
                    main(
                        ifname=sys.argv[1], classes=classes,
                        input_cols=input_cols, target_col=target_col)
                    if len(sys.argv) == 5:
                        experiment = sys.argv[5]
                        main(
                            ifname=sys.argv[1], classes=classes,
                            input_cols=input_cols, target_col=target_col, experiment=experiment)