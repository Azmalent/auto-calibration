from math import radians
from multiprocessing.connection import Client, Listener
from screeninfo import get_monitors
from math_helper import *
import camera_driver
import cv2
import numpy as np
import os
import random

address = ('localhost', 6001)

MAX_ANGLE = 50

def log(message):
    print('[Generator] ' + message)


def load_checkerboard():
    path = os.path.join(os.getcwd(), 'board/checkerboard.png')
    checkerboard = cv2.imread(path)
    checkerboard = cv2.cvtColor(checkerboard, cv2.COLOR_RGB2RGBA)
    return checkerboard

def init_window():
    cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def validate_corners(image, monitor, mat):
    """Validate the positions of the corners"""
    corners = corner_positions(image, mat)

    for corner in corners:
        assert(0 <= corner[0] <= monitor.width)
        assert(0 <= corner[1] <= monitor.height)


def random_scale_and_offsets(image, monitor, mat):
    """Generate random scale and offsets for the checkerboard"""
    corners = corner_positions(image, mat)
    (left, right, top, bottom) = bounds(corners)

    # Random scale
    (width, height) = (right - left, bottom - top)
    max_scale = min(monitor.width / width, monitor.height / height)
    scale = random.uniform(max_scale / 2, max_scale)

    # Apply scale and recalculate bounds
    corners = [(x * scale, y * scale) for (x, y) in corners]
    (left, right, top, bottom) = bounds(corners)

    # Random offsets
    dx_min = int(-left)
    dx_max = int(monitor.width - right)
    dy_min = int(-top)
    dy_max = int(monitor.height - bottom)

    dx = random.uniform(dx_min, dx_max)
    dy = random.uniform(dy_min, dy_max)   
    return (scale, dx, dy)


def transform_image(image, monitor): 
    rx = random_angle(MAX_ANGLE)
    ry = random_angle(MAX_ANGLE)
    rz = random_angle(180)

    dz = np.sqrt(monitor.height**2 + monitor.width**2)

    proj_3d = proj_matrix_3d(monitor)
    rot     = rotation_matrix(rx, ry, rz)
    trans   = translation_matrix_3d(0, 0, dz)
    proj_2d = proj_matrix_2d(monitor, dz)

    mat = proj_2d @ trans @ rot @ proj_3d

    (scale, dx, dy) = random_scale_and_offsets(image, monitor, mat)
    mat = translation_matrix_2d(dx, dy) @ scale_matrix_2d(scale) @ mat

    validate_corners(image, monitor, mat)

    return cv2.warpPerspective(image.copy(), mat, (monitor.width, monitor.height), borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128, 255))


if __name__ == '__main__':
    log('started')

    listener = Listener(address, authkey=b'password')

    monitor = get_monitors()[0]

    init_window()
    checkerboard = load_checkerboard()

    conn = listener.accept()
    camera_client = Client(camera_driver.address, authkey=b'password')
    log('connected to camera driver')

    num_images = 0
    try:
        while True:
            msg = conn.recv()
            if msg == 'next':
                log('received next message')
                transformed_image = transform_image(checkerboard, monitor)

                cv2.imshow('window', transformed_image)
                cv2.waitKey(300) # TODO: wait until image is displayed

                camera_client.send('capture')
            elif msg == 'exit':
                log('received exit message')
                break
    except Exception as e:
        log('error: ' + e.__class__.__name__)
        print(e)
    finally:
        listener.close()
        conn.close()

        camera_client.send('exit')
        camera_client.close()

        cv2.destroyAllWindows()