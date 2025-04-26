import numpy as np

PLOT_STYLES = {
    "amp_layered": "-", 
    "angle_layered": "-",
    "amp_cascade": "-",
    "amp_alternate": "-",
    "amp_cross_mesh": "-",
    "angle_cross_mesh": "-",
    "angle_alternate": "-",
    "angle_cascade": "-",
    "cv": "-",
    "gcv": "-",
    "classical": "-",
}


def feature_scaling(prediction, exact, target_min, target_max):
    """
    Scale the prediction to match the range of exact values.
    Handles cases where the prediction range is degenerate (min == max).

    Parameters:
        prediction (array-like): Predicted values to be scaled.
        exact (array-like): Target values to scale to.

    Returns:
        array-like: Scaled prediction values.
    """
    min_exp = prediction.min()
    max_exp = prediction.max()

    # perform normal feature scaling
    shifted_values = [
        (exp - min_exp) / (max_exp - min_exp) * (target_max - target_min) + target_min
        for exp in prediction
    ]

    return np.array(shifted_values)


def z_score_normalize(values):
    """
    Applies Z-Score normalization to an array of values.

    Parameters:
        values (array-like): The input values to normalize.

    Returns:
        array-like: Z-Score normalized values.
    """
    # convert input to numpy array for convenience
    values = np.array(values)

    # calculate the mean and standard deviation
    mean = np.mean(values)
    std_dev = np.std(values)

    # apply z-score normalization
    z_scores = (values - mean) / std_dev

    return z_scores
