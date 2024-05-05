from utils import translation_matrix_3d
import cv2
import numpy as np


class VirtualCamera:
    def __init__(self, monitor, matrix, camera_dist):
        self.width = monitor.width
        self.height = monitor.height
        self.matrix = matrix

        self.dz = camera_dist * (monitor.width / monitor.width_mm)
        self.nodal_offset = None
        
        # Calculate 3D points of the image plane
        x = np.linspace(-self.width/2, self.width/2, self.width)
        y = np.linspace(-self.height/2, self.height/2, self.height)
        xv, yv = np.meshgrid(x, y)

        X = xv.reshape(-1, 1)
        Y = yv.reshape(-1, 1)
        Z = X * 0
        W = X * 0 + 1

        self.plane = np.concatenate(([X], [Y], [Z], [W]))[:, :, 0]    


    def project(self):
        points = translation_matrix_3d(0, 0, self.dz) @ self.plane

        if self.nodal_offset is not None:
            R, T = self.nodal_offset
            Rt = np.transpose(R)

            P = np.eye(4)
            P[:3,:3] = Rt
            P[:3, 3] = Rt @ -T.reshape(3)

            #P[:3,:3] = R
            #P[:3, 3] = T.reshape(3)

            points = P @ points
        
        X = points[0,:]
        Y = points[1,:]
        Z = points[2,:]

        fx = self.matrix[0,0]
        fy = self.matrix[1,1]
        u0 = self.matrix[0,2]
        v0 = self.matrix[1,2]

        U = fx * (X / Z) + u0
        V = fy * (Y / Z) + v0
        return np.concatenate(([U], [V]))


    def capture(self, image):
        pts2d = self.project()
        xs, ys = np.split(pts2d, 2)
        map_x = xs.reshape(self.height, self.width).astype(np.float32)
        map_y = ys.reshape(self.height, self.width).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
