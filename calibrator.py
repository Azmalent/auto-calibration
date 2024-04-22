from generator import ImageGenerator
from multiprocessing.connection import Client, Listener
from screeninfo import get_monitors
from utils import find_free_port, init_dir
import cv2
import numpy as np
import os
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

BOARD_SIZE = (8, 6)
BOARD_WIDTH, BOARD_HEIGHT = BOARD_SIZE
BOARD_AREA = BOARD_WIDTH * BOARD_HEIGHT

class LensCalibrator():
    def __init__(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((BOARD_AREA, 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:BOARD_WIDTH, 0:BOARD_HEIGHT].T.reshape(-1,2)

        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane

        self.num_captures = 0
        self.gray = None


    def log(self, message):
        print('[Lens Calibrator] ' + message)
    

    def accept_image(self, img):
        self.log('received image')
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(self.gray, BOARD_SIZE, None)

        if success:
            self.log('found corners')
            cv2.imwrite('captures/lens/capture' + str(self.num_captures) + '.png', img)
        
            self.objpoints.append(self.objp)
            corners2 = cv2.cornerSubPix(self.gray, corners, (11, 11), (-1,-1), self.criteria)
            self.imgpoints.append(corners2)
            self.num_captures += 1
        else:
            self.log('failed to find corners')


    def is_done(self):
        return self.num_captures >= 10
    

    def calibrate_lens(self):
        img = cv2.imread('captures/lens/capture0.png')

        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
        np.savetxt('output/camera_matrix.txt', mtx)
        np.savetxt('output/distortion.txt', dist)
        
        errors = []
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            errors.append(cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2))

        mean_error = sum(errors) / len(self.objpoints)
        print("total error: {}".format(mean_error))

        np.savetxt('output/mean_error.txt', [mean_error])
        np.savetxt('output/errors.txt', errors)

        # Undistort image
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)

        # crop the image
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        cv2.imwrite('output/result.png', undistorted)

        return (mtx, new_mtx, dist)


class NodalOffsetCalibrator:
    def __init__(self, mtx, new_mtx, distortion, cam_dist):
        self.matrix = mtx
        self.new_matrix = new_mtx
        self.distortion = distortion
        self.camera_dist = cam_dist
        self.nodal_offset = (0, 0, 0)

        monitor = get_monitors()[0]
        self.gen = ImageGenerator(monitor, cam_dist+1)

        self.num_captures = 0


    def log(self, message):
        print('[Nodal Offset Calibrator] ' + message)


    def accept_image(self, img):
        self.log('received image')

        img = cv2.undistort(img, self.matrix, self.distortion, None, self.new_matrix)
        cv2.imwrite('captures/nodal_offset/capture' + str(self.num_captures) + '.png', img)
        
        img2 = self.gen.next()
        cv2.imwrite('captures/nodal_offset/capture' + str(self.num_captures) + '_virtual.png', img2)

        self.num_captures += 1


    def is_done(self):
        return self.num_captures >= 10


if __name__ == '__main__':
    print('Enter distance from camera to screen in meters: ', end='')
    cam_distance = float( input() )

    init_dir('captures')
    init_dir('captures/lens')
    init_dir('captures/nodal_offset')
    init_dir('output')

    CALIBRATOR_PORT = find_free_port()
    GENERATOR_PORT = find_free_port()
    CAMERA_DRIVER_PORT = find_free_port()

    print('Calibrator port: ' + str(CALIBRATOR_PORT))
    print('Generator port: ' + str(GENERATOR_PORT))
    print('Camera driver port: ' + str(CAMERA_DRIVER_PORT))

    calibrator = LensCalibrator()
    listener = Listener(('localhost', CALIBRATOR_PORT), authkey=b'password')
    
    subprocess.Popen(['python', os.path.join(SCRIPT_DIR, 'generator.py'), 
        '--listener-port', str(GENERATOR_PORT), 
        '--client-port', str(CAMERA_DRIVER_PORT)]) 
    
    subprocess.Popen(['python', os.path.join(SCRIPT_DIR, 'camera_driver.py'),
        '--listener-port', str(CAMERA_DRIVER_PORT),
        '--client-port', str(CALIBRATOR_PORT)])
    
    generator_client = Client(('localhost', GENERATOR_PORT), authkey=b'password')
    calibrator.log('connected to generator')

    conn = listener.accept()

    generator_client.send('next')

    try:
        while True:
            img = conn.recv()
            if img is not None:
                calibrator.accept_image(img)

                if calibrator.is_done():
                    calibrator.log('calibration complete!')

                    if type(calibrator) == LensCalibrator:
                        (mtx, new_mtx, dist) = calibrator.calibrate_lens()
                        calibrator = NodalOffsetCalibrator(mtx, new_mtx, dist, cam_distance)

                        seed = os.getpid()
                        calibrator.gen.random.seed(seed)
                        generator_client.send('change_seed ' + str(seed))
                    else:
                        break

                generator_client.send('next')
    finally:
        listener.close()
        conn.close()

        generator_client.send('exit')
        generator_client.close()