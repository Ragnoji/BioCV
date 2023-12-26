import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import OPTICS


def draw_volume_and_calculate(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    hull = ConvexHull(points)

    for simplex in hull.simplices:
        ax.plot3D(points[simplex, 0], points[simplex, 1], points[simplex, 2], 's-')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
    return hull.volume
