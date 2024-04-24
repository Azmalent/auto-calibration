from datetime import datetime
from generator import ImageGenerator
from multiprocessing.connection import Client, Listener
from screeninfo import get_monitors
from utils import find_free_port, init_dir, translation_matrix_3d
import cv2
import numpy as np
import os
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

SQUARE_SIZE = 128
BOARD_SIZE = (8, 6)

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

class BaseCalibrator():
    def __init__(self):
        self.monitor = get_monitors()[0]

        self.objpoints = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
        self.objpoints[:,:2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1,2)

        self.num_captures = 0

class LensCalibrator(BaseCalibrator):
    def __init__(self):
        super().__init__()

        self.imgpoints = []


    def log(self, message):
        print('[Lens Calibrator] ' + message)
    

    def accept_image(self, img):        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

        if success:
            self.log('found corners')
            cv2.imwrite('captures/lens/capture' + str(self.num_captures) + '.png', img)
        
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), CRITERIA)
            self.imgpoints.append(corners2)
            self.num_captures += 1
        else:
            self.log('failed to find corners')


    def is_done(self):
        return self.num_captures >= 10
    

    def calibrate_lens(self):
        img = cv2.imread('captures/lens/capture' + str(self.num_captures - 1) + '.png')
        h, w = img.shape[:2]

        n = self.num_captures

        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([self.objpoints] * n, self.imgpoints, (w, h), None, None)
        np.savetxt('output/camera_matrix.txt', mtx)
        np.savetxt('output/distortion.txt', dist)
        
        errors = []
        for i in range(n):
            imgpoints2, _ = cv2.projectPoints(self.objpoints, rvecs[i], tvecs[i], mtx, dist)
            errors.append(cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2))

        mean_error = sum(errors) / n
        print("total error: {}".format(mean_error))

        np.savetxt('output/mean_error.txt', [mean_error])
        np.savetxt('output/errors.txt', errors)

        # Undistort image
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)

        # crop the image
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        cv2.imwrite('output/undistorted.png', undistorted)

        return (mtx, new_mtx, dist)


class NodalOffsetCalibrator(BaseCalibrator):
    def __init__(self, mtx, new_mtx, distortion, cam_dist):
        super().__init__()

        self.p_imgpoints = []
        self.v_imgpoints = []

        self.p_matrix = mtx
        self.new_matrix = new_mtx
        self.distortion = distortion
        self.camera_dist = cam_dist
        self.nodal_offset = None

        self.gen = ImageGenerator(self.monitor, cam_dist + 1000)

        w, h = self.monitor.width, self.monitor.height
        f = np.sqrt(w**2 + h**2)
        self.v_matrix = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])


    def log(self, message):
        print('[Nodal Offset Calibrator] ' + message)


    def accept_image(self, img):
        img = cv2.undistort(img, self.p_matrix, self.distortion, None, self.new_matrix)
        img2 = self.gen.next(self.nodal_offset)

        gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret1, corners1 = cv2.findChessboardCorners(gray1, BOARD_SIZE, None)

        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret2, corners2 = cv2.findChessboardCorners(gray2, BOARD_SIZE, None)

        if ret1 and ret2:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1,-1), CRITERIA)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1,-1), CRITERIA)
            #cv2.drawChessboardCorners(img, BOARD_SIZE, corners1, ret1)
            #cv2.drawChessboardCorners(img2, BOARD_SIZE, corners2, ret2)

            cv2.imwrite('captures/nodal_offset/capture' + str(self.num_captures) + '_physical.png', img)
            cv2.imwrite('captures/nodal_offset/capture' + str(self.num_captures) + '_virtual.png', img2)

            self.p_imgpoints.append(corners1)
            self.v_imgpoints.append(corners2)

            if self.is_done():
                self.calibrate_nodal_offset()

            self.num_captures += 1


    def calibrate_nodal_offset(self):
        n = len(self.p_imgpoints)
        (w, h) = self.monitor.width, self.monitor.height

        no_distortion = np.zeros((5, 1))
        _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate([self.objpoints] * n, self.v_pimgpoints, self.p_imgpoints, self.v_matrix, no_distortion, self.p_matrix, no_distortion, (w, h), flags=cv2.CALIB_FIX_INTRINSIC)

        rot = np.array([
            [R[0][0], R[0][1], R[0][2], 0],
            [R[1][0], R[1][1], R[1][2], 0],
            [R[2][0], R[2][1], R[2][2], 0],
            [0, 0, 0, 1]
        ])
        
        trans = translation_matrix_3d(T[0][0], T[1][0], T[2][0])
        self.nodal_offset = (rot, trans)

        np.savetxt('output/nodal_offset_rotation.txt', R)
        np.savetxt('output/nodal_offset_translation.txt', T)
    

    def is_done(self):
        return self.num_captures > 10


if __name__ == '__main__':
    print('Enter distance from camera to screen in mm: ', end='')
    cam_distance = int( input() )

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

    generator_client.send(['next'])

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

                        seed = datetime.now().timestamp()
                        calibrator.gen.random.seed(seed)
                        generator_client.send(['set_seed', seed])
                    else:
                        break

                generator_client.send(['next'])
    finally:
        listener.close()
        conn.close()

        generator_client.send(['exit'])
        generator_client.close()