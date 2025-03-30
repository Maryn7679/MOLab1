import matplotlib.pyplot as plt
import numpy as np
import time


def calculate_objective(x):
    return 10 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def compute_gradient(x):
    return np.array([40*x[0]**3 - 40*x[0]*x[1] + 2*x[0] - 2, 20 * (x[1] - (x[0]**2))])


def compute_second_gradient(x):
    return np.array([[120*x[0]**2 - 40*x[1] + 2, -40*x[0]],
                    [-40*x[0], 20]])


# def calculate_sq_norm(x):
#     return x[0]**2 + x[1]**2
#
#
# def gradient_descent0(x0, step_size, end_condition):
#     results = np.array([])
#     for i in range(10000):
#         gradient_x0 = compute_gradient(x0)
#         x1 = x0 - np.multiply(gradient_x0, step_size)
#         if calculate_objective(x1) - calculate_objective(x0) + epsilon*calculate_sq_norm(gradient_x0) <= 0:
#             if calculate_norm(compute_gradient(x1)) <= end_condition:
#                 break
#             else:
#                 x0 = x1
#                 results = np.append(results, calculate_objective(x0))
#         else:
#             step_size = step_size / 2
#     return x1, results


def newton_minimizer(x0, end_condition):
    results = np.array([])
    directions = np.array([])
    second_derivatives = np.array([])
    for i in range(10000):
        if i % 1000 == 0:
            print(f"Iteration {i}: x = {x0}")
        second_derivative = compute_second_gradient(x0)
        direction = np.linalg.solve(second_derivative, -compute_gradient(x0))
        x1 = x0 + direction
        function_value = calculate_objective(x0)
        if abs(function_value - calculate_objective(x1)) <= end_condition:
            break
        else:
            x0 = x1
            results = np.append(results, function_value)
            directions = np.append(directions, direction)
            second_derivatives = np.append(second_derivatives, second_derivative)
    results = np.append(results, calculate_objective(x0))
    return x1, i, results, directions, second_derivatives


start1 = np.array([2, 4])
start2 = np.array([-2, 10])
epsilon = 0.000001

descent1 = newton_minimizer(start1, epsilon)
print(f"Found x* = {descent1[0]} starting from {start1} in {descent1[1]} iterations")
calculations1 = descent1[2]


descent2 = newton_minimizer(start2, epsilon)
print(f"Found x* = {descent2[0]} starting from {start2} in {descent2[1]} iterations")
calculations2 = descent2[2]

plt.plot(calculations1)
plt.plot(calculations2)
plt.show()
