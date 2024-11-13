import sympy as sy
import numpy as np
from typing import Callable


def partially_derivate(
    function: Callable, all_parameters: list[sy.Symbol], single_parameter: sy.Symbol
) -> Callable:
    """
    Get the derivative of a function with respect to a single parameter.

    Args:
        function (Callable): The function to be evaluated.
        all_parameters (list[sy.Symbol]): A list of all parameters as sympy symbols.
        single_parameter (sy.Symbol): The parameter with respect to which the sensitivity is calculated.

    Returns:
        Callable: The derivative of the function with respect to the single parameter as a function.
    """
    derivative = sy.diff(function(*all_parameters), single_parameter)
    return derivative


def get_sensitivity_for_parameter(
    function: Callable,
    all_parameters: list[sy.Symbol],
    parameter_values: dict[sy.Symbol, float],
    single_parameter: sy.Symbol,
) -> tuple[float]:
    """
    Get the sensitivities of a function with respect to a single parameter.

    Args:
        function (Callable): The function to be evaluated.
        all_parameters (list[sy.Symbol]): A list of all parameters as sympy symbols.
        parameter_values (dict[sy.Symbol, float]): A dictionary containing the values of all parameters.
        single_parameter (sy.Symbol): The parameter with respect to which the sensitivity is calculated.

    Returns:
        dict[str,float]: A dict containing the absolute and relative sensitivity.
    """
    normal_operating_point = function(*all_parameters).subs(parameter_values)
    partial_derivative = partially_derivate(function, all_parameters, single_parameter)
    absolute_sensitivity = partial_derivative.subs(parameter_values)
    relative_sensitivity = (
        absolute_sensitivity
        * parameter_values[single_parameter]
        / normal_operating_point
    )
    sensitivities = {}
    sensitivities["absolute_sensitivity"] = absolute_sensitivity
    sensitivities["relative_sensitivity"] = relative_sensitivity
    return sensitivities


def get_sensitivities(
    function: Callable,
    parameters: list[sy.Symbol],
    parameter_values: dict[sy.Symbol, float],
) -> dict[str, float]:
    """
    Get the sensitivities of a function with respect to all its parameters.
    This function uses sympy for symbolic differentiation - if your function is
    not compatible with sympy, use a numerical approach instead.

    Args:
        function (Callable): The function to be evaluated.
        parameters (list[sy.Symbol]): A list of all the function's parameters as sympy symbols.
        parameter_values (dict[sy.Symbol, float]): A dictionary containing the values of all parameters.
    Returns:
        dict[str,float]: A dict containing the absolute and relative sensitivities for each parameters.

    """
    sensitivities = {}
    for parameter in parameters:
        sensitivities[parameter.__str__()] = get_sensitivity_for_parameter(
            function, parameters, parameter_values, parameter
        )
    return sensitivities


def get_sensitivities_numerical(
    function: Callable,
    parameter_values: dict[str, float],
    interval_width: dict[str, float] or float = 0.05,
    differentiation_interval_values: dict[str, list] = None,
) -> dict[str, float]:
    """
    Get the sensitivities of a function with respect to all its parameters.
    This function uses numerical differentiation - if your function is
    compatible with sympy, use a symbolic approach instead.

    Args:
        function (Callable): The function to be evaluated.
        parameter_values (list): A dict of all the function's parameters and values at NoP.
        interval_width (float or dict): The interval around the normal operating
            point for numerical differentiation. Can be specified as a single float
            (for all parameters) or as a dict with a float for each parameter.
        differentiation_interval_values (list[list]): A list of lists containing the parameter
            interval [low, high] for numerical differentiation. If given, this superseeds the
            interval_width, if None (for any value, the interval_width[p], interval_width['default]
            or plain interval_width (if a float) will be used to set up the low and high value.

    Returns:
        dict[str,float]: A dict containing the absolute and relative sensitivities for each parameters.

    Raises:
        ValueError: If the no valid differentiation_interval or interval width is specified.
        ValueError: If the function evaluation fails for any parameter value.

    """
    normal_operating_point = function(*parameter_values.values())
    sensitivities = {}

    for parameter, normal_value in parameter_values.items():
        try:
            differentiation_interval = differentiation_interval_values[parameter]
        except (KeyError, TypeError):
            differentiation_interval = None

        if not differentiation_interval:
            if isinstance(interval_width, dict):
                try:
                    rel_width = interval_width[parameter]
                except KeyError:
                    try:
                        rel_width = interval_width["default"]
                    except KeyError:
                        raise ValueError(
                            f"Missing differentiation interval for parameter {parameter}. "
                            f"Either specify a single value, or one for every parameter/ a 'default' "
                            f"interval or one for each parameter."
                        )
            else:
                rel_width = interval_width
            differentiation_interval = [
                normal_value * (1 - rel_width),
                normal_value * (1 + rel_width),
            ]
        results = []
        for value in differentiation_interval:
            function_inputs = parameter_values.copy()
            function_inputs[parameter] = value
            try:
                result = function(*function_inputs.values())
            except Error as e:
                raise ValueError(
                    f"Function evaluation with value {value} for parameter {parameter} failed: {e}"
                )
            results.append(result)
        res_diff = results[1] - results[0]
        absolute_sensitivity = res_diff / (
            differentiation_interval[1] - differentiation_interval[0]
        )
        relative_sensitivity = (
            absolute_sensitivity * normal_value / normal_operating_point
        )

        # check if the change is linear from low to normal to high
        try:
            second_derivative = (normal_operating_point - results[0]) / (
                results[1] - normal_operating_point
            )
        except ZeroDivisionError:
            second_derivative = 1

        if np.round(second_derivative, 5) == 1:
            pass
        else:
            print(
                f"the change for parameter {parameter} is not linear, (Nop-low)/(high-norm) = {second_derivative}"
            )
        sensitivities[parameter.__str__()] = {
            "absolute_sensitivity": np.round(absolute_sensitivity),
            "relative_sensitivity": np.round(relative_sensitivity, 2),
        }
    return sensitivities
