import cv2
import numpy as np
from screeninfo import get_monitors
from utils import translation_matrix_3d


class VirtualCamera:
    def __init__(self, monitor, matrix, camera_dist):
        self.width = monitor.width
        self.height = monitor.height
        self.matrix = matrix
        
        dz = camera_dist * (monitor.width / monitor.width_mm)
        self.trans = translation_matrix_3d(0, 0, dz)

        # Plane
        x = np.linspace(-self.width/2, self.width/2, self.width)
        y = np.linspace(-self.height/2, self.height/2, self.height)
        xv, yv = np.meshgrid(x, y)

        X = xv.reshape(-1, 1)
        Y = yv.reshape(-1, 1)
        Z = X * 0
        W = X * 0 + 1

        self.plane = np.concatenate(([X], [Y], [Z], [W]))[:, :, 0]


    def project(self):
        points = self.trans @ self.plane
        xs = points[0,:] / points[2,:]
        ys = points[1,:] / points[2,:]

        X = self.matrix[0,0] * xs + self.matrix[0,2]
        Y = self.matrix[1,1] * ys + self.matrix[1,2]

        return np.concatenate(([X], [Y]))


    def capture(self, image):
        pts2d = self.project()
        xs, ys = np.split(pts2d, 2)
        map_x = xs.reshape(self.height, self.width).astype(np.float32)
        map_y = ys.reshape(self.height, self.width).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
