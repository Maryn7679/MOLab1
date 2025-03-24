import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

random.seed(122)
file = pd.read_csv('weight-height.csv', sep=',')
file["Buffer"] = 1

y = file["Weight"]
X = file[["Buffer", "Height"]]


def calculate_objective0(o):
    return 1/(2*len(y)) * np.linalg.norm(y - X @ o)**2


def compute_gradient0(o):
    return -1/len(y) * X.transpose() @ (y - X @ o)


def gradient_descent0(x0, step_size, end_condition):
    results = np.array([])
    for i in range(500):
        gradient_x0 = compute_gradient0(x0)
        x1 = x0 - np.multiply(gradient_x0, step_size)
        if np.linalg.norm(compute_gradient0(x1)) <= end_condition:
            break
        else:
            x0 = x1
            results = np.append(results, calculate_objective0(x0))
    return np.array(x1), results


def calculate_objective1(o):
    return 1/(2*len(y)) * np.linalg.norm(y - X @ o)**2


def compute_gradient1(o):
    i = random.randrange(len(y))
    return np.array([-(y[i] - o[0] - o[1]*X.loc[i]["Height"]),
                     -X.loc[i]["Height"] * (y[i] - o[0] - o[1]*X.loc[i]["Height"])])


def gradient_descent1(x0, step_size, end_condition):
    results = np.array([])
    for i in range(500):
        gradient_x0 = compute_gradient1(x0)
        x1 = x0 - np.multiply(gradient_x0, step_size)
        if np.linalg.norm(compute_gradient1(x1)) <= end_condition:
            break
        else:
            x0 = x1
            results = np.append(results, calculate_objective1(x0))
    return np.array(x1), results


def calculate_objective2(o):
    fsum = 0
    for q in range(500):
        i = random.randrange(len(y))
        fsum += np.abs(y[i] - o[0] - o[1]*X.loc[i]["Height"])
    return fsum/len(y)


def compute_gradient2(o):
    gsum = np.zeros(2)
    for q in range(500):
        i = random.randrange(len(y))
        if y[i] - o[0] - o[1]*X.loc[i]["Height"] > 0:
            gsum += np.array([-1, -X.loc[i]["Height"]])
        else:
            gsum += np.array([1, X.loc[i]["Height"]])
    return gsum/len(y)


def gradient_descent2(x0, step_size, end_condition):
    results = np.array([])
    for i in range(500):
        gradient_x0 = compute_gradient2(x0)
        x1 = x0 - np.multiply(gradient_x0, step_size)
        if np.linalg.norm(compute_gradient2(x1)) <= end_condition:
            break
        else:
            x0 = x1
            results = np.append(results, calculate_objective2(x0))
    return np.array(x1), results


start = np.array([0, 1]).transpose()
end = 0.0001

L1 = (1/len(y)) * np.linalg.norm(X, 2)**2
step1 = 1/L1

plt.scatter(X["Height"], y)

descent0 = gradient_descent0(start, step1, end)
descent1 = gradient_descent1(start, step1, end)
descent2 = gradient_descent2(start, step1, end)

print(descent0[0])
print(descent1[0])
print(descent2[0])

plt.axline((0, descent0[0].item(0)), (1, descent0[0].item(0) + descent0[0].item(1)))
plt.axline((0, descent1[0].item(0)), (1, descent1[0].item(0) + descent1[0].item(1)))
plt.axline((0, descent2[0].item(0)), (1, descent2[0].item(0) + descent2[0].item(1)))

plt.show()
