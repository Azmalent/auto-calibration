from generator import ImageGenerator, CHECKERBOARD
from screeninfo import get_monitors
from utils import *
import cv2

monitor = get_monitors()[0]
gen = ImageGenerator(monitor, 'extrinsic', 225)
gen.camera_matrix[0:3, 0:3] = np.loadtxt('output/camera_matrix.txt')

RVEC = np.loadtxt('output/rvec.txt')
TVEC = np.loadtxt('output/tvec.txt')

cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

for i in range(1):
    pixels_per_mm = monitor.width / monitor.width_mm

    img = gen.next()

    K = np.loadtxt('output/camera_matrix.txt')
    R = cv2.Rodrigues(RVEC)[0]
    T = np.array([TVEC]).transpose() * pixels_per_mm    

    C = K @ np.hstack((R, T))

    print(C)

    proj_3d = proj_matrix_3d(img.shape[1], img.shape[0])    
    mat = C @ translation_matrix_3d(0, 0, gen.z_dist) @ proj_3d

    img2 = cv2.warpPerspective(img, mat, (monitor.width, monitor.height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 0, 0, 255))

    cv2.imshow('window', img2)
    cv2.waitKey(1000)
