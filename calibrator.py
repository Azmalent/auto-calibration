from multiprocessing.connection import Client, Listener
import camera_driver
import generator
import os
import cv2
import numpy as np
import subprocess

address = ('localhost', 6000)

BOARD_SIZE = (8, 6)
BOARD_WIDTH, BOARD_HEIGHT = BOARD_SIZE
BOARD_AREA = BOARD_WIDTH * BOARD_HEIGHT

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def log(message):
    print('[Calibrator] ' + message)


# TODO: command line args
# opts, args = getopt(sys.argv, 'n:w:', ['wait='])
# for opt, arg in opts:
#     if opt == '-n':
#         num_frames = arg
#     elif opt in ('-w', '--wait'):
#         wait_interval = arg

if __name__ == '__main__':
    log('started')

    listener = Listener(address, authkey=b'password')

    subprocess.Popen(['python', 'generator.py']) 
    subprocess.Popen(['python', 'camera_driver.py'])
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((BOARD_AREA, 3), np.float32)
    objp[:,:2] = np.mgrid[0:BOARD_WIDTH, 0:BOARD_HEIGHT].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    generator_client = Client(generator.address, authkey=b'password')
    log('connected to generator')

    conn = listener.accept()

    generator_client.send('next')

    gray = None
    num_successful = 0
    try:
        while True:
            img = conn.recv()
            if img is not None:
                log('received image')

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                success, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

                if success:
                    log('found corners')
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), criteria)
                    imgpoints.append(corners2)
                    num_successful += 1
                else:
                    log('failed to find corners')

                if num_successful >= 10: # TODO: finish conditions
                    break

                generator_client.send('next')
    finally:
        listener.close()
        conn.close()

        generator_client.send('exit')
        generator_client.close()

    if not os.path.isdir('output'):
        os.mkdir('output')
        
    # images = glob.glob('captures/*.png')
    # images.sort(key=lambda x: float(re.findall("(\d+)",x)[0]))

    # for fname in images:
    #     img = cv2.imread(fname)
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     cv2.imshow('capture', gray)
    #     cv2.waitKey(500)
        

    #     if success:
    #         objpoints.append(objp)
    #         corners2 = cv2.cornerSubPix(gray,corners, (11, 11), (-1,-1), criteria)
    #         imgpoints.append(corners2)
            
    #         cv2.drawChessboardCorners(img, BOARD_SIZE, corners2, success)
    #         cv2.imshow('capture', img)
    #         cv2.waitKey(500)


    
    img = cv2.imread('captures/capture0.png')

    ret, mat, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savetxt('output/camera_matrix.txt', mat)
    np.savetxt('output/distortion.txt', dist)

    mean_error = 0
    for num_successful in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[num_successful], rvecs[num_successful], tvecs[num_successful], mat, dist)
        error = cv2.norm(imgpoints[num_successful], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: {}".format(mean_error/len(objpoints)) )

    # Undistort image
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat, dist, (w,h), 1, (w,h))

    dst = cv2.undistort(img, mat, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('output/result.png', dst)