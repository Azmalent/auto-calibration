from getopt import getopt
from math import radians
from multiprocessing.connection import Client, Listener
from random import Random
from screeninfo import get_monitors
from utils import *
import cv2
import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

CHECKERBOARD = cv2.imread( os.path.join(SCRIPT_DIR, 'board/checkerboard.png') )
CHECKERBOARD = cv2.cvtColor(CHECKERBOARD, cv2.COLOR_RGB2RGBA)


def init_window(monitor):
    cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('window', -monitor.width, 0)
    cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


class ImageGenerator():
    MAX_ANGLE = 50
    BACKGROUND_COLOR = (0, 255, 0, 255)

    def __init__(self, monitor, distance_from_camera):
        self.monitor = monitor
        self.random = Random()

        pixels_per_meter = monitor.width / monitor.width_mm * 1000
        self.z_dist = distance_from_camera * pixels_per_meter


    def log(self, message):
        print('[Generator] ' + message)


    def next(self): 
        """
        Generates the next image.
        """
        rx = self.random_angle(ImageGenerator.MAX_ANGLE)
        ry = self.random_angle(ImageGenerator.MAX_ANGLE)
        rz = self.random_angle(180)

        proj_3d = proj_matrix_3d(self.monitor)
        rot     = rotation_matrix(rx, ry, rz)
        trans   = translation_matrix_3d(0, 0, self.z_dist)
        proj_2d = proj_matrix_2d(self.monitor, np.sqrt(self.monitor.height**2 + self.monitor.width**2))

        mat = proj_2d @ trans @ rot @ proj_3d

        (dx, dy) = self.random_offsets(mat)

        mat = translation_matrix_2d(dx, dy) @ mat

        image = cv2.warpPerspective(CHECKERBOARD.copy(), mat, (self.monitor.width, self.monitor.height), borderMode=cv2.BORDER_CONSTANT, borderValue=ImageGenerator.BACKGROUND_COLOR)
        
        return image


    def random_angle(self, max_degrees):
        """Generate random angle in range [-max_degrees, max_degrees]"""
        degrees = self.random.uniform(-max_degrees, max_degrees)
        return radians(degrees)


    def random_offsets(self, mat):
        """Generate random offsets for the checkerboard"""
        corners = corner_positions(CHECKERBOARD, mat)
        (left, right, top, bottom) = bounds(corners)

        dx_min = int(-left)
        dx_max = int(self.monitor.width - right)
        dy_min = int(-top)
        dy_max = int(self.monitor.height - bottom)

        dx = self.random.uniform(dx_min, dx_max)
        dy = self.random.uniform(dy_min, dy_max)   
        return (dx, dy)


if __name__ == '__main__':
    listener_port = None
    client_port = None

    opts, args = getopt(sys.argv[1:], '', ['listener-port=', 'client-port='])
    for opt, arg in opts:
        if opt == '--listener-port':
            listener_port = int(arg)
        elif opt == '--client-port':
            client_port = int(arg)

    assert listener_port is not None
    assert client_port is not None

    listener = Listener(('localhost', listener_port), authkey=b'password')

    monitor = get_monitors()[0]
    gen = ImageGenerator(monitor, 1)

    init_window(monitor)

    conn = listener.accept()
    camera_client = Client(('localhost', client_port), authkey=b'password')

    try:
        while True:
            msg = conn.recv()
            if msg == 'next':
                gen.log('received next message')
                img = gen.next()

                cv2.imshow('window', img)
                cv2.waitKey(1000)
                
                camera_client.send('capture')
            elif msg.startswith('change_seed '):
                seed = int(msg[len('change_seed '):])
                gen.log('changing seed to ' + str(seed))
                gen.random.seed(seed)
            elif msg == 'exit':
                gen.log('received exit message')
                break
    except Exception as e:
        gen.log('error: ' + e.__class__.__name__)
        print(e)
    finally:
        listener.close()
        conn.close()

        camera_client.send('exit')
        camera_client.close()

        cv2.destroyAllWindows()