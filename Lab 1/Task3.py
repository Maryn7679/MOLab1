import matplotlib.pyplot as plt
import numpy as np


def calculate_objective1(x):
    return 1/2 * (x - np.log(1 + x))**2


def calculate_objective2(x):
    return 1/2 * (x - np.log(2 + x))**2


def compute_gradient1(x):
    return x + np.log(1+x)/(1+x) - np.log(1+x) - x/(1+x)


def compute_gradient2(x):
    return x + np.log(2+x)/(2+x) - np.log(2+x) - x/(2+x)


def gradient_descent1(x0, step_size, end_condition):
    results = np.array([])
    for i in range(100):
        gradient_x0 = compute_gradient1(x0)
        x1 = x0 - np.multiply(gradient_x0, step_size)
        if np.abs(compute_gradient1(x1)) <= end_condition:
            break
        else:
            x0 = x1
            results = np.append(results, calculate_objective1(x0))
    return x1, results


def gradient_descent2(x0, step_size, end_condition):
    results = np.array([])
    for i in range(100):
        gradient_x0 = compute_gradient2(x0)
        x1 = x0 - np.multiply(gradient_x0, step_size)
        if np.abs(compute_gradient2(x1)) <= end_condition:
            break
        else:
            x0 = x1
            results = np.append(results, calculate_objective2(x0))
    return x1, results


start = 2

ax = plt.gca()
x_axis = np.linspace(0, 2, 100)
y1 = x_axis
y2 = np.log(1 + x_axis)
y3 = np.log(2 + x_axis)
plt.plot(x_axis, y1)
plt.plot(x_axis, y2)
plt.plot(x_axis, y3)
plt.show()

y2 = (x_axis - np.log(1 + x_axis))**2 / 2
plt.plot(x_axis, y2)
dy2 = x_axis + np.log(1+x_axis)/(1+x_axis) - np.log(1+x_axis) - x_axis/(1+x_axis)
plt.plot(x_axis, dy2)
plt.axhline(0.62)
plt.show()

y3 = (x_axis - np.log(2 + x_axis))**2 / 2
plt.plot(x_axis, y3)
dy3 = x_axis + np.log(2+x_axis)/(2+x_axis) - np.log(2+x_axis) - x_axis/(2+x_axis)
plt.plot(x_axis, dy3)
plt.axhline(0.49)
plt.show()

L1 = 1/0.62
descent1 = gradient_descent1(start, L1, 0.000001)
calculations1 = descent1[1]
print("Intersection point for 1: " + f'{descent1[0]:.5f}')
plt.plot(calculations1)

L2 = 1/0.49
descent2 = gradient_descent2(start, L2, 0.000001)
calculations2 = descent2[1]
print("Intersection point for 2: " + f'{descent2[0]:.5f}')
plt.plot(calculations2)

plt.show()
