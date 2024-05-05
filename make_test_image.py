import cv2
from generator import CHECKERBOARD
import numpy as np
from screeninfo import get_monitors
from utils import proj_matrix_3d, translation_matrix_3d

# Project checkerboard onto 1920x1080 screen from 1 meter away
# Needed to measure SQUARE_SIZE

monitor = get_monitors()[0]
pixels_per_mm = monitor.width / monitor.width_mm

f = np.sqrt(monitor.width**2 + monitor.height**2)

K = np.zeros((3, 4))
K[:3, :3] = np.array([
    [f, 0, monitor.width / 2],
    [0, f, monitor.height / 2],
    [0, 0, 1]
])

M = K @ translation_matrix_3d(0, 0, 500 * pixels_per_mm) @ proj_matrix_3d(CHECKERBOARD.shape[1], CHECKERBOARD.shape[0])
TEST_IMG = cv2.warpPerspective(CHECKERBOARD, M, (1920, 1080))
cv2.imwrite('test.png', TEST_IMG)