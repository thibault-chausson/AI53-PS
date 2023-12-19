# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:22:22 2021

@author: LAURI
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class KMeans:
    def __init__(self, K, minv, maxv, inputs):
        self.K = K
        plt.xlim(minv, maxv)
        plt.ylim(minv, maxv)
        self.Centers = np.random.uniform(minv, maxv, (K, 2))
        self.InitialCenters = self.Centers.copy()
        self.Inputs = inputs
        self.Size = len(self.Inputs)
        self.C = np.zeros(self.Size, dtype=int)
        self.display()

    def next(self):
        self.assignToCenter()
        self.updateCenters()
        self.display()

    def assignToCenter(self):
        pass

    def updateCenters(self, lr=0.1):
        pass

    def J(self):
        sd = 0.0
        for i, x in enumerate(self.Inputs):
            c = self.Centers[self.C[i]]
            sd += self.d(x, c)
        return sd

    def animate(self, i):
        self.next()
        scat.set_offsets(self.Centers)
        first_scat.set_offsets(self.InitialCenters)
        return scat, first_scat

    def display(self):
        for k, p in enumerate(self.Centers):
            print(k, p)
        print('J:', self.J())

    def d(self, x, y):
        return np.linalg.norm(x - y)


# Dataset
data = np.array([(1, 10), (1.5, 2), (1, 6), (2, 1.5), (2, 10), (3, 2.5), (3, 6), (4, 2)])
classes = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int)

# Display the dataset
color_names = ['red', 'blue']
colors = [color_names[c - 1] for c in classes]

fig = plt.figure()  # initialise la figure
x, y = zip(*data)
plt.scatter(x, y, c=colors)
plt.show()

first_scat = plt.scatter([], [], c='blue', s=80, marker='s', edgecolor='black')
scat = plt.scatter([], [], c='green', s=160, marker='s', edgecolor='black')

kmeans = KMeans(2, -11, 11, data)
kmeans.animate(0)

ani = animation.FuncAnimation(fig, kmeans.animate, interval=2000, repeat=True)
plt.show()
