import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar


def calculate_objective(x):
    return 10 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def compute_gradient(x):
    return np.array([40*x[0]**3 - 40*x[0]*x[1] + 2*x[0] - 2, 20 * (x[1] - (x[0]**2))])


def update_hessian(h, diff, delta_g):
    diff = diff[:, np.newaxis]
    return h + ((diff @ diff.transpose()) / (delta_g.transpose() @ diff))

# def update_hessian(h, diff, delta_x):
#     diff = diff[:, np.newaxis]
#     return h + ((diff @ diff.transpose()) / (diff.transpose() @ delta_x))


def f(a, x, y, d, c):
    return 10*(y + a*c - (x + a*d)**2)**2 + (1 - x - a*d)**2


def calculate_step(x, direction):
    fun = lambda a: f(a, x[0], x[1], direction[0], direction[1])
    # print(minimize_scalar(fun, bounds=[0, 100], method='bounded').x)
    return minimize_scalar(fun, bounds=[0, 100], method='bounded').x


def quasi_newton_minimizer(x0, end_condition):
    results = np.array([])
    directions = np.array([])
    hessians = np.array([])

    hessian = np.identity(2)
    for i in range(10000):
        gradient = compute_gradient(x0)
        direction = np.linalg.solve(hessian, -gradient)
        x1 = x0 + direction * calculate_step(x0, direction)
        function_value = calculate_objective(x0)
        delta_gradient = compute_gradient(x1) - gradient
        if abs(function_value - calculate_objective(x1)) <= end_condition:
            break
        else:
            delta_x = x1 - x0
            difference = delta_x - hessian @ delta_gradient
            hessian = update_hessian(hessian, difference, delta_gradient)

            # delta_x = x1 - x0
            # difference = delta_gradient - hessian @ delta_x
            # hessian = update_hessian(hessian, difference, delta_x)

            x0 = x1
            results = np.append(results, function_value)
            directions = np.append(directions, direction)
            hessians = np.append(hessians, hessian)
    results = np.append(results, calculate_objective(x0))
    return x1, i, results, directions, hessians


start1 = np.array([2, 4])
start2 = np.array([-2, 10])
epsilon = 0.000001


descent1 = quasi_newton_minimizer(start1, epsilon)
print(f"Found x* = {descent1[0]} starting from {start1} in {descent1[1]} iterations")
calculations1 = descent1[2]

descent2 = quasi_newton_minimizer(start2, epsilon)
print(f"Found x* = {descent2[0]} starting from {start2} in {descent2[1]} iterations")
calculations2 = descent2[2]

plt.plot(calculations1)
plt.plot(calculations2)
plt.show()
