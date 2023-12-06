import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def draw_volume_and_calculate(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # представим двухмерное облако точек лежащее в 0 координате Z
    # p1 = np.zeros((50, 3))
    # print(p1)
    # p1[:, :2] = np.random.normal(loc=1, scale=2, size=(50, 2))
    # print(p1)

    # представим двухмерное облако точек лежащее в 5 координате Z, типа раздвинули области друг от друга
    # p2 = np.ones((40, 3)) * 5
    # print(p2)
    # p2[:, :2] = np.random.normal(loc=0.9, scale=1.5, size=(40, 2))
    # print(p2)

    # создаём пространство
    # points = np.vstack([p1, p2])
    # print(points)
    hull = ConvexHull(points)
    # считаем объём :D
    print('vol', hull.volume)

    edges = list(zip(*points))
    for i in hull.simplices:
        plt.plot(points[i, 0], points[i, 1], points[i, 2], 'r-')
        ax.plot(edges[0], edges[1], edges[2], 'bo')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
    return hull.volume
