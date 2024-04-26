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

    def __init__(self, monitor, mode = 'lens', mm_from_camera = 1000):
        self.monitor = monitor
        self.mode = mode
        
        w, h = monitor.width, monitor.height
        f = np.sqrt(w**2 + h**2)
        self.camera_matrix = np.array([
            [f, 0, w / 2, 0],
            [0, f, h / 2, 0],
            [0, 0, 1, 0]
        ])

        self.random = Random()

        pixels_per_mm = monitor.width / monitor.width_mm
        self.z_dist = mm_from_camera * pixels_per_mm


    def log(self, message):
        print('[Generator] ' + message)


    def next(self, nodal_offset = None): 
        """
        Generates the next image.
        """
        mat = None
        if self.mode == 'lens':
            mat = self.matrix_lens()
        elif self.mode == 'nodal':
            mat = self.matrix_nodal()
        elif self.mode == 'virtual':
            mat = self.matrix_virtual(nodal_offset)

        return cv2.warpPerspective(CHECKERBOARD.copy(), mat, (self.monitor.width, self.monitor.height), borderMode=cv2.BORDER_CONSTANT, borderValue=ImageGenerator.BACKGROUND_COLOR)


    # For lens calibration
    # Random 3D rotation + random 2D offset
    def matrix_lens(self):
        proj_3d = proj_matrix_3d(CHECKERBOARD.shape[1], CHECKERBOARD.shape[0])

        rx = self.random_angle(ImageGenerator.MAX_ANGLE)
        ry = self.random_angle(ImageGenerator.MAX_ANGLE)
        rz = self.random_angle(180)
        rot = rotation_matrix(rx, ry, rz)

        trans = translation_matrix_3d(0, 0, self.z_dist)

        mat = self.camera_matrix @ trans @ rot @ proj_3d

        (dx, dy) = self.random_offsets(mat)
        offset_2d = translation_matrix_2d(dx, dy)

        return offset_2d @ mat


    # For nodal offset calibration (screen)
    # Random 2D offset, rotation and scale
    def matrix_nodal(self):
        a = self.random_angle(180)
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

        corners = corner_positions(CHECKERBOARD, rot)
        (left, right, top, bottom) = bounds(corners)

        # Random scale
        (width, height) = (right - left, bottom - top)
        max_scale = min(self.monitor.width / width, self.monitor.height / height)
        
        scale = self.random.uniform(max_scale / 2, max_scale)
        scale_mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])

        mat = rot @ scale_mat
        (dx, dy) = self.random_offsets(mat)
        return translation_matrix_2d(dx, dy) @ mat
    
    
    # For nodal offset calibration (virtual camera)
    # Random 2D + 3D offset on Z axis
    def matrix_virtual(self, nodal_offset = None):
        nodal = self.matrix_nodal()

        proj_3d = proj_matrix_3d(self.monitor.width, self.monitor.height)
        
        mat = translation_matrix_3d(0, 0, self.z_dist)

        if nodal_offset is not None:
            (R, T) = nodal_offset
            mat = translation_matrix_3d(T[0], T[1], T[2]) @ R @ mat

        return self.camera_matrix @ mat @ proj_3d @ nodal


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
                elif command == 'set_mode':
                    gen.mode = args[0]
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