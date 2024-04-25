from contextlib import closing
import numpy as np
import os
import socket

def init_dir(dir):
    """
    Initializes a directory by removing all files within it if it already exists, or creating it if it doesn't.
    """
    if os.path.isdir(dir):
        for file_name in os.listdir(dir):
            file = dir + "/" + file_name
            if os.path.isfile(file):
                os.remove(file)
    else:
        os.mkdir(dir)


def find_free_port():
    """Find a free port on localhost"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def normalize(p):
    """Normalize a point in homogeneous coordinates"""
    return p[:-1] / p[-1]


def corner_positions(image, mat):
    """Calculate the position of the corners of the checkerboard after transformation"""
    (height, width) = image.shape[:2]
    return [normalize(mat.dot([x, y, 1])) for x in [0, width] for y in [0, height]]


def bounds(corners):
    """Calculate the bounds of a tetragon given its corner positions"""
    (xs, ys) = zip(*corners)
    return (min(xs), max(xs), min(ys), max(ys))


def proj_matrix_3d(monitor):
    """2D -> 3D projection matrix"""
    return np.array([   [1, 0, -monitor.width/2],
                        [0, 1, -monitor.height/2],
                        [0, 0, 1],
                        [0, 0, 1]])


def proj_matrix_2d(fx, fy, cx, cy):
    """3D -> 2D projection matrix"""
    return np.array([   [fx, 0, cx, 0],
                        [0, fy, cy, 0],
                        [0, 0,  1,  0]])


def rotation_matrix(rx, ry, rz):
    """Rotation matrix"""
    RX = np.array([ [1, 0, 0, 0],
                    [0, np.cos(rx), -np.sin(rx), 0],
                    [0, np.sin(rx), np.cos(rx), 0],
                    [0, 0, 0, 1]])
    
    RY = np.array([ [np.cos(ry), 0, -np.sin(ry), 0],
                    [0, 1, 0, 0],
                    [np.sin(ry), 0, np.cos(ry), 0],
                    [0, 0, 0, 1]])
    
    RZ = np.array([ [np.cos(rz), -np.sin(rz), 0, 0],
                    [np.sin(rz), np.cos(rz), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    return RX.dot(RY).dot(RZ)


def translation_matrix_2d(dx, dy):
    """2D translation matrix"""
    return np.array([   [1, 0, dx],
                        [0, 1, dy],
                        [0, 0, 1]])


def translation_matrix_3d(dx, dy, dz):
    """3D translation matrix"""
    return np.array([   [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])