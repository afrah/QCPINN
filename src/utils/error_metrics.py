import numpy as np


def lp_error(pred, exact, text, logger, p):
    num = np.sum(np.abs(pred - exact) ** p)
    denum = np.sum(np.abs(exact) ** p)
    if denum == 0.0:
        denum = 1.0
        text = text + " (Absolute (denominator is zero))"
    result = ((num / denum) ** (1 / p)) * 100
    logger.print("%s  : %5.3e " % (text, result))
    return result
