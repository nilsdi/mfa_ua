import pandas as pd
import numpy as np


from typing import Callable


def get_XYZ(
    function: Callable, x_range: tuple, y_range: tuple, n_steps: int = 100
) -> tuple[np.array]:
    """
    Returns XYZ values for a contour plot, based of a function that
    only takes x and y as arguments (usually requires a wrapper function).

    Args:
        function (Callable): The function to be plotted - f(x, y)
        x_range (tuple): The range of x values to be plotted (low, high)
        y_range (tuple): The range of y values to be plotted (low, high)
        n_steps (int, optional): The number of steps in the x and y range. Defaults to 100.

    Returns:
        tuple(np.array): X, Y, Z values for the contour plot
    """
    X = np.linspace(x_range[0], x_range[1], n_steps)
    Y = np.linspace(y_range[0], y_range[1], n_steps)
    Z = np.zeros((n_steps, n_steps))

    for ix, x in enumerate(X):
        for iy, y in enumerate(Y):
            Z[ix, iy] = function(x, y)

    return X, Y, Z


def save_XYZ(X: np.array, Y: np.array, Z: np.array, filename: str) -> None:
    """
    Save X, Y, Z to a dataframe (csv file).
    """
    df = pd.DataFrame(Z, index=X, columns=Y)
    df.to_csv(filename)
    return
