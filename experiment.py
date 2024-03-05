from math import radians
from cv2 import BORDER_CONSTANT
from math_helper import *
from multiprocessing.connection import Client
from screeninfo import get_monitors
import cv2
import numpy as np
import os

def transform_image(image, monitor, rx, ry, rz): 
    (height, width) = image.shape[:2]

    dz = np.sqrt(monitor.width**2 + monitor.height**2)

    trans_2d = np.matrix([  [1, 0, (monitor.width - width) / 2],
                            [0, 1, (monitor.height - height) / 2],
                            [0, 0, 1]])
    
    A1 = proj_matrix_3d(monitor)
    R  = rotation_matrix(rx, ry, rz)
    T  = translation_matrix_3d(0, 0, dz)
    A2 = proj_matrix_2d(monitor, dz)

    mat = A2.dot(T).dot(R).dot(A1).dot(trans_2d)

    return cv2.warpPerspective(image.copy(), mat, (monitor.width, monitor.height), borderMode=BORDER_CONSTANT, borderValue=(128, 128, 128, 255))


if __name__ == '__main__':
    address = ('localhost', 6000)
    conn = Client(address, authkey=b'password')

    num_frames = 90 // 5
    wait_interval = 500

    monitor = get_monitors()[0]

    path = os.path.join(os.getcwd(), 'board/checkerboard.png')
    checkerboard = cv2.imread(path)
    checkerboard = cv2.cvtColor(checkerboard, cv2.COLOR_RGB2RGBA)

    cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for i in range(num_frames):
        ry = radians((i+1) * 5)
        rx = radians((i+1) * 5)

        transformed_image = transform_image(checkerboard, monitor, rx, ry, 0)

        cv2.imshow('window', transformed_image)
        
        cv2.waitKey(wait_interval // 2)
        conn.send('capture')
        cv2.waitKey(wait_interval // 2)

    cv2.waitKey(500)

    conn.send('exit')
    conn.close()