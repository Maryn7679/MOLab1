import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file = pd.read_csv('weight-height.csv', sep=',')
file["Buffer"] = 1

y = file["Weight"]
X = file[["Buffer", "Height"]]


def calculate_objective(o):
    return 1/(2*len(y)) * np.linalg.norm(y - X @ o)**2


def compute_gradient(o):
    return -1/len(y) * X.transpose() @ (y - X @ o)


def gradient_descent(x0, step_size, end_condition):
    results = np.array([])
    for i in range(500):
        gradient_x0 = compute_gradient(x0)
        x1 = x0 - np.multiply(gradient_x0, step_size)
        if np.linalg.norm(compute_gradient(x1)) <= end_condition:
            break
        else:
            x0 = x1
            results = np.append(results, calculate_objective(x0))
    return np.array(x1), results


start = np.array([0, 1]).transpose()
end = 0.0001

L1 = (1/len(y)) * np.linalg.norm(X, 2)**2
step1 = 1/L1
descent0 = gradient_descent(start, step1, end)
calculations0 = descent0[1]
print(descent0[0])
print(calculate_objective(descent0[0]))
plt.plot(calculations0)
plt.show()
