import matplotlib.pyplot as plt
import numpy as np
import time


A = np.array([[9.25336081, 4.16813642, 52.33516583, 49.88931588, 24.87415869],
              [63.40408605, 18.09228273, 76.47216652, 39.21961636, 62.96827423],
              [85.34801857, 77.68434276, 90.02636313, 67.76068249, 60.45494377],
              [75.66350896, 47.62214697, 22.78026612, 47.2833045, 67.52843101],
              [50.5551226, 64.16985232, 56.34440244, 58.82215568, 29.84572218],
              [90.87166047, 27.33525755, 87.71712115, 50.90259541, 98.43768442],
              [34.53877287, 49.54361297, 20.48232987, 78.57303301, 35.9609915]])
b = np.array([19.20927724, 45.17305068, 0.43302717, 37.00013531, 72.33081754, 46.95261308, 70.60775347])


def calculate_objective(x):
    return 1/(2*A.shape[0]) * np.linalg.norm(A @ x - b, 2)


def compute_gradient(x):
    return 1/(2 * A.shape[0]) * (A.transpose() @ (A @ x - b))/np.linalg.norm(A @ x - b, 2)


def gradient_descent(x0, step_size, end_condition):
    results = np.array([])
    for i in range(50):
        gradient_x0 = compute_gradient(x0)
        x1 = x0 - np.multiply(gradient_x0, step_size)
        if np.linalg.norm(compute_gradient(x1)) <= end_condition:
            break
        else:
            x0 = x1
            results = np.append(results, calculate_objective(x0))
    return x1, results


start1 = np.array([1, 5, 9, 7, 3]).transpose()
start2 = np.array([1, 1, 1, 2, 1]).transpose()
end = 0.0001

L1 = 1/A.shape[0] * np.linalg.norm(A)
L2_max = 1/A.shape[0] * (np.linalg.norm(A.transpose() @ A) * np.linalg.norm(start2) + np.linalg.norm(A.transpose()@b))
print(L2_max)

step1 = 1/L1
step2 = 1/400

descent1 = gradient_descent(start1, step1, end)
calculations1 = descent1[1]

descent2 = gradient_descent(start2, step2, end)
calculations2 = descent2[1]

plt.plot(calculations1)
plt.show()
plt.plot(calculations2)
plt.show()
