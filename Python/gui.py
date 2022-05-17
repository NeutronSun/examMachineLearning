#pip install matplotlib
from matplotlib import cm, pyplot as plt

import math

import numpy as np 

#make a class
class GUI:

    def inizializePlot(dataset, toAnalize, w1, w2, b):
        x, y, z = [], [], []
        for point in dataset:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = np.arange(1, 2.1, 0.25)
        Y = np.arange(3, 9, 0.25)
        X, Y = np.meshgrid(X, Y)
        zf = np.array(1/(1+np.exp(-w1*X-w2*Y-b)))
        Z = zf.reshape(X.shape)
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                      linewidth=0, antialiased=True, alpha=0.711)
        ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=2)
        ax.set_xlim([1, 2])
        ax.set_ylim([3, 9])
        ax.set_zlim([-0.1, 1.1])
        #fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.ylabel('Weight')
        plt.xlabel('Heigth')
        plt.show()

    def prova(dataset, w1, w2, b):
        
        plt.show()
