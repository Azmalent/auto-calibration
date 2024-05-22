import cv2
import numpy as np
from screeninfo import get_monitors
from utils import proj_matrix_3d, translation_matrix_3d

# Project checkerboard onto 1920x1080 screen from 1 meter away
monitor = get_monitors()[1]
pixels_per_mm = monitor.width / monitor.width_mm

f = np.sqrt(monitor.width**2 + monitor.height**2)

K = np.zeros((3, 4))
K[:3, :3] = np.loadtxt('output/camera_matrix.txt')
# K[:3, :3] = np.array([
#     [f, 0, monitor.width / 2],
#     [0, f, monitor.height / 2],
#     [0, 0, 1]
# ])

IMAGE = cv2.imread('captures/nodal_offset/capture0_original.png')

f = K[0][0]
pixels_per_mm = f / 282
dz = 383 * pixels_per_mm

M = K @ translation_matrix_3d(0, 0, dz) @ proj_matrix_3d(IMAGE.shape[1], IMAGE.shape[0])
TEST_IMG = cv2.warpPerspective(IMAGE, M, (1920, 1080))
cv2.imwrite('test.png', TEST_IMG)