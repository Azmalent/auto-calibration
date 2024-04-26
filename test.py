from generator import ImageGenerator
from screeninfo import get_monitors
from utils import *
import cv2

monitor = get_monitors()[0]
gen = ImageGenerator(monitor, 'nodal', 225)
gen.camera_matrix[0:3, 0:3] = np.loadtxt('output/camera_matrix.txt')

cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

for i in range(5):
    img = gen.next()

    proj_3d = proj_matrix_3d(monitor.width, monitor.height)    
    mat = gen.camera_matrix @ translation_matrix_3d(0, 0, gen.z_dist) @ proj_3d

    img2 = cv2.warpPerspective(img, mat, (monitor.width, monitor.height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 0, 255))

    cv2.imshow('window', img2)
    cv2.waitKey(250)
