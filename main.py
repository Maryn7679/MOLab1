import matplotlib.pyplot as plt
import numpy as np


def calculate_objective(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def compute_gradient(x):
    return np.array([400 * ((x[0]**3) - (x[0] * x[1])) + 2*x[0] - 2, 200 * (x[1] - (x[0]**2))])


def calculate_norm(x):
    return np.sqrt(x[0]**2 + x[1]**2)


def calculate_sq_norm(x):
    return x[0]**2 + x[1]**2


def gradient_descent(x0, step_size, end_condition):
    results = np.array([])
    for i in range(10000):
        gradient_x0 = compute_gradient(x0)
        x1 = x0 - np.multiply(gradient_x0, step_size)
        if calculate_objective(x1) - calculate_objective(x0) + epsilon*calculate_sq_norm(gradient_x0) <= 0:
            if calculate_norm(compute_gradient(x1)) <= end_condition:
                break
            else:
                x0 = x1
                results = np.append(results, calculate_objective(x0))
        else:
            step_size = step_size / 2
    return x1, results


start = np.array([-2, 2])
step = 0.1
end = 0.0001
epsilon = 0

descent = gradient_descent(start, step, end)
print(descent[0])
calculations = descent[1]

ax = plt.gca()
plt.plot(calculations)
ax.set_ylim(0, 7)
plt.show()
