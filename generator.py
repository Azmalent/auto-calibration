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

    def __init__(self, monitor, mm_from_camera = 1000):
        self.monitor = monitor
        
        w, h = monitor.width, monitor.height
        f = np.sqrt(w**2 + h**2)
        self.set_camera_matrix(f, f, w / 2, h / 2)

        self.random = Random()
        self.apply_2d_offset = True

        pixels_per_mm = monitor.width / monitor.width_mm
        self.z_dist = mm_from_camera * pixels_per_mm


    def log(self, message):
        print('[Generator] ' + message)


    def set_camera_matrix(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


    def next(self, nodal_offset = None): 
        """
        Generates the next image.
        """
        rx = self.random_angle(ImageGenerator.MAX_ANGLE)
        ry = self.random_angle(ImageGenerator.MAX_ANGLE)
        rz = self.random_angle(180)
        
        trans = translation_matrix_3d(0, 0, self.z_dist)
        rot   = rotation_matrix(rx, ry, rz)

        mat = trans @ rot

        if nodal_offset is not None:
            (R, T) = nodal_offset
            print(mat)
            mat = translation_matrix_3d(T[0], T[1], T[2]) @ R @ mat
            print(mat)

        mat = proj_matrix_2d(self.fx, self.fy, self.cx, self.cy) @ mat @ proj_matrix_3d(self.monitor)

        # 2D offsets
        (dx, dy) = (None, None)
        if self.apply_2d_offset and nodal_offset is None: 
            (dx, dy) = self.random_offsets(mat)
        else:
            (dx, dy) = self.center_offset(mat)

        mat = translation_matrix_2d(dx, dy) @ mat

        image = cv2.warpPerspective(CHECKERBOARD.copy(), mat, (self.monitor.width, self.monitor.height), borderMode=cv2.BORDER_CONSTANT, borderValue=ImageGenerator.BACKGROUND_COLOR)
        
        return image


    def random_angle(self, max_degrees):
        """Generate random angle in range [-max_degrees, max_degrees]"""
        degrees = self.random.uniform(-max_degrees, max_degrees)
        return radians(degrees)


    def center_offset(self, mat):
        """Move the 2d object to the middle of the screen"""
        corners = corner_positions(CHECKERBOARD, mat)
        (left, right, top, bottom) = bounds(corners)

        dx = (self.monitor.width - right - left) / 2 
        dy = (self.monitor.height - bottom - top) / 2
        return (dx, dy)

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
    gen = ImageGenerator(monitor)

    init_window(monitor)

    conn = listener.accept()
    camera_client = Client(('localhost', client_port), authkey=b'password')

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
                elif command == 'set_camera_matrix':
                    mtx = args[0]
                    gen.set_camera_matrix(mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])
                elif command == 'disable_2d_offset':
                    gen.apply_2d_offset = False
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