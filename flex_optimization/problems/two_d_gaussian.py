import numpy as np


def two_d_gaussian(args, sigma_x: float = 1, sigma_y: float = 1, pre_factor: float = 1, x_o: float = 0, y_o: float = 0):
    """ Two-dimensional Gaussian function

    Parameters
    ----------
    args: array [2]
        [x, y]
    sigma_x: float
        standard deviation in x direction
    sigma_y: float
        standard deviation in y direction
    pre_factor: float
        pre-factor
    x_o: float
        x center location
    y_o: float
        y center location

    Returns
    -------
    return: float
        z value

    """
    x = args[0]
    y = args[1]
    return pre_factor*np.exp(-((x-x_o)**2/(2*sigma_x**2)+(y-y_o)**2/(2*sigma_y**2)))
