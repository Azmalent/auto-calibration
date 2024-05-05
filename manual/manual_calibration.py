from glob import glob
from screeninfo import get_monitors
import cv2
import numpy as np
import os

MAX_ALLOWED_ERROR = 0.999

MONITOR = get_monitors()[0]
PIXELS_PER_MM = MONITOR.width / MONITOR.width_mm

SQUARE_SIZE_MM = 29
SQUARE_SIZE_PIXELS = SQUARE_SIZE_MM * PIXELS_PER_MM

BOARD_SIZE = (8, 6)
objpoints = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objpoints[:,:2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1,2) * SQUARE_SIZE_PIXELS

captures = sorted( glob('captures/*.jpg') )

filenames = []
images = []
imgpoints = []

for i in range(len(captures)):
    filename = captures[i]
    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        filenames.append(filename)
        imgpoints.append(corners)
        images.append(img)
    else:
        print('Failed to find corners in file ' + filename + '. Deleting')
        os.remove(filename)

n = len(images)
h, w = images[0].shape[:2]
rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objpoints] * n, imgpoints, (w, h), None, None)

print('RMS: ' + str(rms))
np.savetxt('camera_matrix.txt', mtx)
np.savetxt('distortion.txt', dist)

errors = []
for i in range(n):
    imgpoints2, _ = cv2.projectPoints(objpoints, rvecs[i], tvecs[i], mtx, dist)
    errors.append(cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2))

mean_error = sum(errors) / n
print("total error: {}".format(mean_error))

np.savetxt('mean_error.txt', [mean_error])

# Delete files where error is too high
for i in range( len(errors) ):
    err = errors[i]
    if err > MAX_ALLOWED_ERROR:
        filename = filenames[i]
        print('Error too high in file ' + filename + ' (' + str(err) + '). Deleting')
        os.remove(filename)