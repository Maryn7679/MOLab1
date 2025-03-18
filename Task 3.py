import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

random.seed(122)
file = pd.read_csv('weight-height.csv', sep=',')
file["Buffer"] = 1

y = file["Weight"]
X = file[["Buffer", "Height"]]


def calculate_objective(o):
    fsum = 0
    for i in range(len(y)):
        fsum += np.abs(y[i] - o[0] - o[1]*X.loc[i]["Height"])
    return fsum/len(y)


def compute_gradient(o):
    gsum = np.zeros(2)
    for i in range(len(y)):
        if y[i] - o[0] - o[1]*X.loc[i]["Height"] > 0:
            gsum += np.array([-1, -X.loc[i]["Height"]])
        else:
            gsum += np.array([1, X.loc[i]["Height"]])
    return gsum/len(y)


def gradient_descent(x0, step_size, end_condition):
    results = np.array([])
    for i in range(50):
        gradient_x0 = compute_gradient(x0)
        x1 = x0 - np.multiply(gradient_x0, step_size)
        print(x1)
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
