import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fomlads.data.external import standard_scaler

from fomlads.evaluate.eval_classification import two_class_cf_matrix, roc , roc_auc, two_class_f1_score

from fomlads.model.classification import project_data
from fomlads.model.classification import maximum_separation_projection
from fomlads.model.classification import fisher_linear_discriminant_projection


from fomlads.plot.exploratory import plot_scatter_array_classes
from fomlads.plot.exploratory import plot_class_histograms
from fomlads.plot.evaluations import plot_roc

def fisher_main(train_inputs, train_targets, test_inputs, test_targets):
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

    classes = ['0','1']
    project_and_histogram_data(train_inputs, train_targets, method='fisher', classes=classes)

    #calculate weights
    weights = get_projection_weights(train_inputs, train_targets, method='fisher')

    #project test set to 1-D
    projected_inputs = project_data(test_inputs, weights)

    #threshold
    threshold = -0.54339884
    y_pred = np.empty(2000)
    for i in range(len(projected_inputs)):
        if projected_inputs[i] < threshold:
            y_pred[i] = classes[0]
        else:
            y_pred[i] = classes[1]

    #graph 1-D projection
    project_and_histogram_data(test_inputs, y_pred, method='fisher', classes=classes)

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

    #ROC
    construct_and_plot_roc(test_inputs, y_pred, method='fisher', colour='b')

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
    if method == 'fisher':
        weights = fisher_linear_discriminant_projection(inputs, targets)
    else:
        raise ValueError("Unrecognised projection method")
    return weights