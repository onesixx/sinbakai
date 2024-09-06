"""
나맘의 perceptron을 구현해보자
https://www.youtube.com/watch?v=YODTXF9OIiw
"""
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

def step_function(value):
    if value < THRES:
        return 0
    return 1

def gen_training_data(num_points):
    # num_points =6
    x1 = np.random.rand(num_points)
    x2 = np.random.rand(num_points)
    labels =  ((x1+x2) >1).astype(int)
    training_data_set = [((x1[i], x2[i]), labels[i]) for i in range(num_points)]
    return training_data_set

THRES = .5
w = np.array([.3, .9])
lr = .1
num_points =100
epoch = 10
training_set = gen_training_data(num_points)

###### ------ training_set check
print(training_set[0:5])


fig, ax = plt.subplots()
ax.set_ylim(-.1, 1.1)
ax.set_xlim(-.1, 1.1)
ax.set_aspect('equal', adjustable='box')
for x, y in training_set:
    if y == 1:
        ax.scatter(x[0], x[1], c='b', marker='x')
    else:
        ax.scatter(x[0], x[1], c='g', marker='o')
plt.show()


linear_space = np.linspace(0, 1, 50)
for i in range(epoch):
    cnt = 0
    for x , y in training_set:
        clear_output(wait=True)

        u = sum(x*w)
        y_hat = step_function(u)   # np.dot(w, x)
        error = y - y_hat
        for index, value in enumerate(x):
            w[index] = w[index]  + lr * (y - y_hat) * value

        for xs, ys in training_set[0:cnt]:
            fig, ax = plt.subplots()
            ax.set_ylim(-.1, 1.1)
            ax.set_xlim(-.1, 1.1)
            ax.set_aspect('equal', adjustable='box')
            if ys ==1:
                if y == 1:
                    ax.scatter(x[0], x[1], c='b', marker='x')
                else:
                    ax.scatter(x[0], x[1], c='g', marker='o')
        yy  =
