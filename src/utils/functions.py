import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def shift_target_to_output_domain(target, output_min, output_max):
    return target * (output_max - output_min) + output_min
