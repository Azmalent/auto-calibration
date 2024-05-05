from datetime import datetime
from generator import ImageGenerator
from multiprocessing.connection import Client, Listener
from screeninfo import get_monitors
from utils import find_free_port, init_dir
from vcam import VirtualCamera
import argparse
import cv2
import numpy as np
import os
import subprocess


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MONITOR = get_monitors()[0]
MM_PER_PIXEL = MONITOR.width_mm / MONITOR.width
PIXELS_PER_MM = MONITOR.width / MONITOR.width_mm

# The size of a square projection after translating by 1 meter
SQUARE_SIZE = 50.5
BOARD_SIZE = (8, 6)

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)


def parse_args():
    parser = argparse.ArgumentParser(prog='calibrator')
    parser.add_argument('--mode', choices=['full', 'intrinsic', 'extrinsic'], default='full')
    parser.add_argument('-n', '--num_captures', type=int, required=True)
    parser.add_argument('-d', '--distance', type=int, required=True)
    
    return parser.parse_args()


def init_directories():
    init_dir('captures')
    init_dir('captures/lens')
    init_dir('captures/nodal_offset')
    
    if not os.path.isdir('output'):
        os.mkdir('output')


def init_network():
    calibrator_port = find_free_port()
    generator_port = find_free_port()
    camera_driver_port = find_free_port()

    print('Calibrator port: ' + str(calibrator_port))
    print('Generator port: ' + str(generator_port))
    print('Camera driver port: ' + str(camera_driver_port))
    
    listener = Listener(('localhost', calibrator_port), authkey=b'password')
    
    subprocess.Popen(['python', os.path.join(SCRIPT_DIR, 'generator.py'), 
        '--listener-port', str(generator_port), 
        '--client-port', str(camera_driver_port)]) 
    
    subprocess.Popen(['python', os.path.join(SCRIPT_DIR, 'camera_driver.py'),
        '--listener-port', str(camera_driver_port),
        '--client-port', str(calibrator_port)])
    
    generator_client = Client(('localhost', generator_port), authkey=b'password')

    return (listener, generator_client)


def sync_rngs(rng, client):
    seed = datetime.now().timestamp()
    rng.seed(seed)
    client.send(['set_seed', seed])


class AbstractCalibrator():
    def __init__(self, num_snapshots):
        self.monitor = MONITOR
        
        self.imgpoints = []

        self.objpoints = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
        self.objpoints[:,:2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1,2) * SQUARE_SIZE

        self.num_captures = 0
        self.required_num_captures = num_snapshots
        
    def is_done(self):
        return self.num_captures >= self.required_num_captures


class LensCalibrator(AbstractCalibrator):
    def log(self, message):
        print('[Lens Calibrator] ' + message)
    

    def accept_image(self, img):      
        self.log('accepting image #' + str(self.num_captures + 1))

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

    def calibrate_lens(self):
        img = cv2.imread('captures/lens/capture' + str(self.num_captures - 1) + '.png')
        h, w = img.shape[:2]

        n = self.num_captures

        rmse, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([self.objpoints] * n, self.imgpoints, (w, h), None, None)
        np.savetxt('output/camera_matrix.txt', mtx)
        np.savetxt('output/distortion.txt', dist)
        np.savetxt('output/rmse.txt', [rmse])
        
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
        np.savetxt('output/optimal_camera_matrix.txt', new_mtx)

        undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)

        # crop the image
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        cv2.imwrite('output/undistorted.png', undistorted)
        
        self.log('calibration complete!')

        return (mtx, new_mtx, dist)


class NodalOffsetCalibrator(AbstractCalibrator):
    def __init__(self, mtx, new_mtx, distortion, cam_dist, num_captures):
        super().__init__(num_captures)

        self.matrix = mtx
        self.new_matrix = new_mtx
        self.distortion = distortion

        # TODO: pass original image along with camera image through sockets???
        self.gen = ImageGenerator(self.monitor)
        self.vcam = VirtualCamera(self.monitor, mtx, cam_dist)
        self.v_imgpoints = []


    def log(self, message):
        print('[Nodal Offset Calibrator] ' + message)


    def accept_image(self, img):
        self.log('accepting image #' + str(self.num_captures + 1))

        original_img = self.gen.next()
        physical_img = cv2.undistort(img, self.matrix, self.distortion, None, self.new_matrix)
        virtual_img = self.vcam.capture(original_img)
        
        cv2.imwrite('captures/nodal_offset/capture' + str(self.num_captures) + '_original.png', original_img)
        cv2.imwrite('captures/nodal_offset/capture' + str(self.num_captures) + '_physical.png', physical_img)
        cv2.imwrite('captures/nodal_offset/capture' + str(self.num_captures) + '_virtual.png',  virtual_img)

        gray1 = cv2.cvtColor(physical_img, cv2.COLOR_BGR2GRAY)
        ret1, corners1 = cv2.findChessboardCorners(gray1, BOARD_SIZE, None)

        gray2 = cv2.cvtColor(virtual_img, cv2.COLOR_BGR2GRAY)
        ret2, corners2 = cv2.findChessboardCorners(gray2, BOARD_SIZE, None)

        if ret1 and ret2:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1,-1), CRITERIA)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1,-1), CRITERIA)

            self.imgpoints.append(corners1)
            self.v_imgpoints.append(corners2)

            self.num_captures += 1


    def calibrate_nodal_offset(self):
        n = len(self.imgpoints)
        (w, h) = self.monitor.width, self.monitor.height

        no_distortion = np.zeros((5, 1))
        rmse, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate([self.objpoints] * n, self.v_imgpoints, self.imgpoints, self.matrix, no_distortion, self.matrix, no_distortion, (w, h), flags=cv2.CALIB_FIX_INTRINSIC)

        np.savetxt('output/nodal_offset_rotation.txt', R)
        np.savetxt('output/nodal_offset_translation.txt', T)
        np.savetxt('output/nodal_offset_rmse.txt', [rmse])

        self.vcam.nodal_offset = (R, T.flatten())
        self.calculate_error()

        self.log('calibration complete!')

    
    def calculate_error(self):
        errors = []

        for i in range(self.num_captures):
            physical_img = cv2.imread('captures/nodal_offset/capture' + str(i) + '_physical.png')

            original_img = cv2.imread('captures/nodal_offset/capture' + str(i) + '_original.png')
            virtual_img = self.vcam.capture(original_img)

            cv2.imwrite('captures/nodal_offset/capture' + str(i) + '_virtual_corrected.png',  virtual_img)

            gray1 = cv2.cvtColor(physical_img, cv2.COLOR_BGR2GRAY)
            ret1, corners1 = cv2.findChessboardCorners(gray1, BOARD_SIZE, None)

            gray2 = cv2.cvtColor(virtual_img, cv2.COLOR_BGR2GRAY)
            ret2, corners2 = cv2.findChessboardCorners(gray2, BOARD_SIZE, None)

            if ret1 and ret2:
                corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1,-1), CRITERIA)
                corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1,-1), CRITERIA)

                errors.append(cv2.norm(corners1, corners2, cv2.NORM_L2) / len(corners2))
            else:
                raise RuntimeError('failed to find corners for nodal offset error calculation')
            
        mean_error = sum(errors) / len(errors)
        np.savetxt('output/nodal_offset_error.txt', [mean_error])
        self.log('nodal offset error: ' + str(mean_error))


if __name__ == '__main__':
    args = parse_args()
    
    calibrator = None
    (mtx, new_mtx, dist) = None, None, None

    if args.mode != 'extrinsic':
        calibrator = LensCalibrator(args.num_captures)
    else:
        mtx = np.loadtxt('output/camera_matrix.txt')
        new_mtx = np.loadtxt('output/optimal_camera_matrix.txt')
        dist = np.loadtxt('output/distortion.txt')
        calibrator = NodalOffsetCalibrator(mtx, new_mtx, dist, args.distance, args.num_captures)

    init_directories()
    (listener, generator_client) = init_network()
    calibrator.log('connected to generator')

    conn = listener.accept()
    
    if type(calibrator) == NodalOffsetCalibrator:        
        sync_rngs(calibrator.gen.random, generator_client)

    generator_client.send(['next'])

    try:
        while True:
            img = conn.recv()
            if img is not None:
                calibrator.accept_image(img)

                if calibrator.is_done():
                    if type(calibrator) == LensCalibrator:
                        (mtx, new_mtx, dist) = calibrator.calibrate_lens()
                        if args.mode == 'full':
                            calibrator = NodalOffsetCalibrator(mtx, new_mtx, dist, args.distance, args.num_captures)
                            sync_rngs(calibrator.gen.random, generator_client)
                        else:
                            break
                    elif type(calibrator) == NodalOffsetCalibrator:
                        calibrator.calibrate_nodal_offset()
                        break

                generator_client.send(['next'])
    except Exception as e:
        print('Error: ' + e.__class__.__name__)
        print(e)
    finally:
        listener.close()
        conn.close()

        generator_client.send(['exit'])
        generator_client.close()