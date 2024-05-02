from utils import translation_matrix_3d
import cv2
import numpy as np


class VirtualCamera:
    def __init__(self, monitor, matrix, camera_dist):
        self.width = monitor.width
        self.height = monitor.height
        self.matrix = matrix
        self.nodal_offset = None
        
        # Calculate 3D points of the image plane
        x = np.linspace(-self.width/2, self.width/2, self.width)
        y = np.linspace(-self.height/2, self.height/2, self.height)
        xv, yv = np.meshgrid(x, y)

        X = xv.reshape(-1, 1)
        Y = yv.reshape(-1, 1)
        Z = X * 0
        W = X * 0 + 1

        plane = np.concatenate(([X], [Y], [Z], [W]))[:, :, 0]
        
        dz = camera_dist * (monitor.width / monitor.width_mm)
        trans = translation_matrix_3d(0, 0, dz)

        self.points = trans @ plane


    # TODO: nodal offset
    def project_2d(self):
        fx = self.matrix[0, 0]
        fy = self.matrix[1, 1]
        u0 = self.matrix[0, 2]
        v0 = self.matrix[1, 2]

        points = self.points

        if self.nodal_offset is not None:
            R, T = self.nodal_offset
            points = translation_matrix_3d(T[0], T[1], T[2]) @ R @ points

        X, Y, Z = points[0,:], points[1,:], points[2,:]
        return np.concatenate(([fx * (X / Z) + u0], [fy * (Y / Z) + v0]))


    def capture(self, image):
        pts2d = self.project_2d()
        xs, ys = np.split(pts2d, 2)
        map_x = xs.reshape(self.height, self.width).astype(np.float32)
        map_y = ys.reshape(self.height, self.width).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
