
import numpy as np

PLOT_STYLES = {
    "amp_layered": "-",      # Solid line
    "angle_layered": "-",    # Solid line
    "amp_cascade": "-",    # Solid line
    "amp_alternate": "-",    # Solid line
    "amp_cross_mesh": "-",     # Solid line
    "angle_cross_mesh": "-",   # Solid line
    "angle_alternate": "-",  # Solid line
    "angle_cascade": "-",  # Solid line
    "cv": "-",              # Solid line
    "gcv": "-",             # Solid line
    "classical": "-"        # Solid line
}

def feature_scaling(prediction, exact , target_min , target_max):
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

    # Perform normal feature scaling
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
    # Convert input to numpy array for convenience
    values = np.array(values)
    
    # Calculate the mean and standard deviation
    mean = np.mean(values)
    std_dev = np.std(values)
    
    # Apply Z-Score normalization
    z_scores = (values - mean) / std_dev
    
    return z_scores
