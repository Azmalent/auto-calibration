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
    cv2.moveWindow('window', monitor.x, monitor.y)
    cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


class ImageGenerator():
    MAX_ANGLE = 50
    BACKGROUND_COLOR = (0, 255, 0, 255)

    def __init__(self, monitor):
        self.monitor = monitor
        
        self.random = Random()
        
        w, h = monitor.width, monitor.height
        f = np.sqrt(w**2 + h**2)
        self.camera_matrix = np.array([
            [f, 0, w / 2, 0],
            [0, f, h / 2, 0],
            [0, 0, 1, 0]
        ])

        self.proj_3d = proj_matrix_3d(CHECKERBOARD.shape[1], CHECKERBOARD.shape[0])

        self.pixels_per_mm = monitor.width / monitor.width_mm
        self.trans = translation_matrix_3d(0, 0, 750 * self.pixels_per_mm)


    def log(self, message):
        print('[Generator] ' + message)


    def next(self): 
        """Generate the next image."""
        mat = self.random_matrix()
        return cv2.warpPerspective(CHECKERBOARD, mat, (self.monitor.width, self.monitor.height), borderMode=cv2.BORDER_CONSTANT, borderValue=ImageGenerator.BACKGROUND_COLOR)


    # For lens calibration
    # Random 3D rotation + random 2D offset
    def random_matrix(self):
        """Generate random 3D rotation and random 2D offset"""
        degrees = self.random.uniform(40, 50)
        axis = self.random_unit_vector()

        #rx = self.random_angle(ImageGenerator.MAX_ANGLE)
        #ry = self.random_angle(ImageGenerator.MAX_ANGLE)
        #rz = self.random_angle(180)
        #rot = rotation_matrix(rx, ry, rz)
        rot = rotation_matrix_from_axis(axis, np.radians(degrees))

        if self.random.random() > 0.5:
            rot = rot @ rotation_matrix(0, 0, np.radians(90))

        # TODO: flag to control dz randomization
        trans = translation_matrix_3d(0, 0, self.random.uniform(500, 1000) * self.pixels_per_mm)

        mat = self.camera_matrix @ trans @ rot @ self.proj_3d

        (dx, dy) = self.random_offsets(mat)
        offset_2d = translation_matrix_2d(dx, dy)

        return offset_2d @ mat



    def random_angle(self, max_degrees):
        """Generate random angle in range [-max_degrees, max_degrees]"""
        degrees = self.random.uniform(-max_degrees, max_degrees)
        return radians(degrees)
    
    def random_unit_vector(self):
        """Generate random unit vector"""
        phi = self.random.uniform(0, np.pi * 2)
        costheta = self.random.uniform(-1, 1)

        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return (x, y, z)
    

    def random_offsets(self, mat):
        """Generate random offsets for the checkerboard"""
        corners = corner_positions(CHECKERBOARD, mat)
        (left, right, top, bottom) = bounds(corners)

        dx_min = int(-left)
        dx_max = int(self.monitor.width - right)
        dy_min = int(-top)
        dy_max = int(self.monitor.height - bottom)

        #TODO fix error
        dx = max([self.random.uniform(dx_min, dx_max) for _ in range(1)], key=abs)
        dy = max([self.random.uniform(dy_min, dy_max) for _ in range(1)], key=abs)

        return (dx, dy)


if __name__ == '__main__':
    listener_port = None
    client_port = None
    monitor_id = 0

    opts, args = getopt(sys.argv[1:], '', ['listener-port=', 'client-port=', 'monitor='])
    for opt, arg in opts:
        if opt == '--listener-port':
            listener_port = int(arg)
        elif opt == '--client-port':
            client_port = int(arg)
        elif opt == '--monitor':
            monitor_id = int(arg)

    assert listener_port is not None
    assert client_port is not None

    listener = Listener(('localhost', listener_port), authkey=b'password')

    monitor = get_monitors()[monitor_id]
    gen = ImageGenerator(monitor)

    init_window(monitor)

    conn = listener.accept()
    camera_client = Client(('localhost', client_port), authkey=b'password')
    gen.log('connected to camera driver')

    try:
        while True:
            msg = conn.recv()
            if msg is not None:
                command, args = msg[0], msg[1:]
                if command == 'next':
                    img = gen.next()

                    cv2.imshow('window', img)
                    cv2.waitKey(300)
                    
                    camera_client.send('capture')
                elif command == 'set_seed':
                    gen.random.seed(args[0])
                elif command == 'exit':
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