import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fomlads.data.external import import_for_classification

from fomlads.model.classification import project_data
from fomlads.model.classification import maximum_separation_projection
from fomlads.model.classification import fisher_linear_discriminant_projection

from fomlads.plot.exploratory import plot_scatter_array_classes
from fomlads.plot.exploratory import plot_class_histograms
from fomlads.plot.evaluations import plot_roc

def main(ifname, input_cols=None, target_col=None, classes=None):
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
    plot_scatter_array_classes(inputs, targets, field_names=field_names, 
        classes=classes)
    project_and_histogram_data(inputs, targets, method='maximum_separation',
        classes=classes)
    project_and_histogram_data(inputs, targets, method='fisher',
        classes=classes)
    fig, ax = construct_and_plot_roc(
        inputs, targets, method='maximum_separation', colour='g')
    construct_and_plot_roc(
        inputs, targets, method='fisher', colour='b', fig_ax=(fig,ax))
    ax.legend(["maximum_separation", "fisher"])       
    plt.show()

def project_and_histogram_data(
        inputs, targets, method, title=None, classes=None):
    """
    Takes input and target data for classification and projects
    this down onto 1 dimension according to the given method,
    then histograms the projected data.

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1
    """
    weights = get_projection_weights(inputs, targets, method)
    projected_inputs = project_data(inputs, weights)
    ax = plot_class_histograms(projected_inputs, targets)
    # label x axis
    ax.set_xlabel(r"$\mathbf{w}^T\mathbf{x}$")
    ax.set_title("Projected Data: %s" % method)
    if not classes is None:
        ax.legend(classes)

def construct_and_plot_roc(
        inputs, targets, method='maximum_separation', **kwargs):
    """
    Takes input and target data for classification and projects
    this down onto 1 dimension according to the given method,
    then plots roc curve for the data.

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1
    """
    weights = get_projection_weights(inputs, targets, method)
    projected_inputs = project_data(inputs, weights)
    new_ordering = np.argsort(projected_inputs)
    projected_inputs = projected_inputs[new_ordering]
    targets = np.copy(targets[new_ordering])
    N = targets.size
    num_neg = np.sum(1-targets)
    num_pos = np.sum(targets)
    false_positive_rates = np.empty(N)
    true_positive_rates = np.empty(N)
    for i, w0 in enumerate(projected_inputs):
        false_positive_rates[i] = np.sum(1-targets[i:])/num_neg
        true_positive_rates[i] = np.sum(targets[i:])/num_pos
    fig, ax = plot_roc(
        false_positive_rates, true_positive_rates, **kwargs)
    return fig, ax

def get_projection_weights(inputs, targets, method):
    """
    Takes input and target data for classification and projects
    this down onto 1 dimension according to the given method


    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1
    returns
    -------
    weights - the projection vector
    """
    if len(np.unique(targets)) > 2:
        raise ValueError("This method only supports data with two classes")
    if method == 'maximum_separation':
        weights = maximum_separation_projection(inputs, targets)
    elif method == 'fisher':
        weights = fisher_linear_discriminant_projection(inputs, targets)
    else:
        raise ValueError("Unrecognised projection method")
    return weights

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