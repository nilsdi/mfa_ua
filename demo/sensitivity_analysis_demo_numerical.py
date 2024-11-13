# %%
from typing import Callable
import numpy as np

from mfa_ua.sensitivity_analysis.get_sensitivities import get_sensitivities_numerical


if __name__ == "__main__":
    parameter_values = {"x": 1, "y": 2, "z": 3, "w": 4}

    def test_function(x, y, z, w):
        return (x * y + x**z) * w

    sensitivities = get_sensitivities_numerical(test_function, parameter_values)
    for key, item in sensitivities.items():
        print(f"{key}: {item}")

# %%
