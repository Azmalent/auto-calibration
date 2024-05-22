from math import sqrt
from utils import translation_matrix_3d, proj_matrix_3d
import cv2
import numpy as np
import os

# From tracker height and monitor frame width
DY_CORRECTION = 20

class VirtualCamera:
    def __init__(self, monitor, matrix, camera_dist):
        self.width = monitor.width
        self.height = monitor.height

        self.matrix = np.zeros((3, 4))
        self.matrix[:3, :3] = matrix
        
        if os.path.isfile('tracker_positions.txt'):
            self.log('using tracker positions to measure distance')
            with open('tracker_positions.txt') as f:
                split = [line.split() for line in f.readlines()]
                
                # parse and convert to mm
                points = [
                    [float(xi) * 1000 for xi in split[0]],
                    [float(xi) * 1000 for xi in split[1]],
                ]

                # translate from top to the middle of the screen
                dy = monitor.height_mm / 2 + DY_CORRECTION
                points[1][1] -= dy

                distance_mm = sqrt( sum([(points[0][i] - points[1][i]) ** 2 for i in range(3)]) )
                pixels_per_mm = matrix[0][0] / camera_dist
                self.dz = distance_mm * pixels_per_mm
        else:
            self.log('no tracker data found, defaulting to using focus distance as dz')
            self.dz = matrix[0][0]

        self.nodal_offset = None


    def log(self, message):
        print('[Virtual Camera] ' + message)


    def capture(self, image):
        h, w = image.shape[:2]

        M = translation_matrix_3d(0, 0, self.dz)

        if self.nodal_offset is not None:
            R, T = self.nodal_offset

            P = np.eye(4)
            P[:3,:3] = R
            P[:3, 3] = T.reshape(3)

            M = P @ M
        
        M = self.matrix @ M @ proj_matrix_3d(w, h)
        return cv2.warpPerspective(image, M, (w, h))
