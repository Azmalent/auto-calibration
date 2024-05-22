from calibrator import BOARD_SIZE, CRITERIA, SQUARE_SIZE
from screeninfo import get_monitors
from vcam import VirtualCamera
import cv2
import numpy as np

monitor = get_monitors()[0]
mtx = np.loadtxt('output/camera_matrix.txt')
vcam = VirtualCamera(monitor, mtx, 382)

objpts = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objpts[:,:2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1,2) * SQUARE_SIZE

p_imgpts = []
v_imgpts = []

n = 10
for i in range(n):
    physical_img = cv2.imread('captures/nodal_offset/capture' + str(i) + '_physical.png')
    original_img = cv2.imread('captures/nodal_offset/capture' + str(i) + '_original.png')
    virtual_img  = vcam.capture(original_img)

    cv2.imwrite('captures/nodal_offset/capture' + str(i) + '_virtual.png',  virtual_img)
    
    gray1 = cv2.cvtColor(physical_img, cv2.COLOR_BGR2GRAY)
    ret1, corners1 = cv2.findChessboardCorners(gray1, BOARD_SIZE, None)

    gray2 = cv2.cvtColor(virtual_img, cv2.COLOR_BGR2GRAY)
    ret2, corners2 = cv2.findChessboardCorners(gray2, BOARD_SIZE, None)

    if ret1 and ret2:
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1,-1), CRITERIA)
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1,-1), CRITERIA)

        physical_corners = cv2.drawChessboardCorners(physical_img, BOARD_SIZE, corners1, ret1)
        cv2.imwrite('captures/nodal_offset/capture' + str(i) + '_physical_corners.png', physical_corners)

        virtual_corners = cv2.drawChessboardCorners(virtual_img, BOARD_SIZE, corners2, ret2)
        cv2.imwrite('captures/nodal_offset/capture' + str(i) + '_virtual_corners.png', virtual_corners)

        p_imgpts.append(corners1)
        v_imgpts.append(corners2)
    else:
        raise RuntimeError('failed to find corners')

(w, h) = monitor.width, monitor.height

no_dist = np.zeros((5, 1))
rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate([objpts] * n, v_imgpts, p_imgpts, mtx, no_dist, mtx, no_dist, (w, h), flags=cv2.CALIB_FIX_INTRINSIC+cv2.CALIB_FIX_K3)
print(rms)

np.savetxt('output/nodal_offset_rotation.txt', R)
np.savetxt('output/nodal_offset_translation.txt', T)

vcam.nodal_offset = (R, T.flatten())

for i in range(10):
    original_img = cv2.imread('captures/nodal_offset/capture' + str(i) + '_original.png')
    corrected_img = vcam.capture(original_img)

    cv2.imwrite('captures/nodal_offset/capture' + str(i) + '_virtual_corrected.png', corrected_img)

