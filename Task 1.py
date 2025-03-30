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
    for i in range(10000):
        if i % 1000 == 0:
            print(f"Iteration {i}: x = {x0}")
        direction = np.linalg.solve(compute_second_gradient(x0), -compute_gradient(x0))
        x1 = x0 + direction
        function_value = calculate_objective(x0)
        if abs(function_value - calculate_objective(x1)) <= end_condition:
            break
        else:
            x0 = x1
            results = np.append(results, function_value)
    results = np.append(results, calculate_objective(x0))
    return x1, results, i


start = np.array([-2, 10])
epsilon = 0.000001

start_time = time.time()
descent = newton_minimizer(start, epsilon)
exec_time = time.time() - start_time
print(f"Execution time is {exec_time} seconds")
print(f"Found x* = {descent[0]} in {descent[2]} iterations")
calculations = descent[1]

plt.plot(calculations)
plt.show()
