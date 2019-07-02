import meep as mp
import numpy as np


def create_vertices(ampl, periodicity, thickness, resolution, sizex, y=False, matrix=None):
    # create vertices for sinusoidal grating with given parameters
    # if y, rotate it 90 degrees
    # if matrix is not None, additionally transform geometry
    freq = 1 / periodicity
    vertices = []
    aux_vert = []
    dx = 2 / resolution

    def f(x): return ampl / 2 * np.cos(2 * np.pi * x * freq)

    for i in np.arange(-sizex / 2 - dx, sizex / 2 + dx + dx, dx):
        x_coord = i
        y_coord = f(i)
        if matrix is not None:
            trfrmd = np.dot(matrix, [x_coord, y_coord])
            x_coord = trfrmd[0]
            y_coord = trfrmd[1]
        if y:
            vertices.append(mp.Vector3(y_coord, x_coord))
        else:
            vertices.append(mp.Vector3(x_coord, y_coord))

    for i in np.arange(sizex / 2 + dx, -sizex / 2 - dx - dx, -dx):
        x_coord = i
        y_coord = f(i) + thickness
        if matrix is not None:
            trfrmd = np.dot(matrix, [x_coord, y_coord])
            x_coord = trfrmd[0]
            y_coord = trfrmd[1]
        if y:
            aux_vert.append(mp.Vector3(y_coord, x_coord))
        else:
            aux_vert.append(mp.Vector3(x_coord, y_coord))

    return vertices + aux_vert
