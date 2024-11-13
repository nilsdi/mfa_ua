# %%
import matplotlib.pyplot as plt
import numpy as np
from mfa_ua.sensitivity_analysis.contour_plot import get_XYZ


if __name__ == "__main__":
    parameter_values = {"x": 1, "y": 2, "z": 3, "w": 4}

    def f2(x, y, z, w):
        return (x + y + x / z) * w**2

    def wrapped_f2(x, z):
        return f2(x, parameter_values["y"], z, parameter_values["w"])

    X, Y, Z = get_XYZ(wrapped_f2, (0.1, 2), (0.1, 5))
    print(f"range of Z: {Z.min()} to {Z.max()}")
    levels = np.linspace(0, 90, 20)
    fig = plt.contour(X, Y, Z, cmap="viridis", levels=levels)
    cbar = plt.colorbar()
    plt.clabel(fig, inline=True)
    plt.show()

# %%
