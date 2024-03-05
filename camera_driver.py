from cv2 import VideoCapture
from multiprocessing.connection import Client, Listener
import calibrator
import cv2
import os

address = ('localhost', 6002)

def log(message):
    print('[Camera driver] ' + message)


def init_directory():
    if os.path.isdir('captures'):
        for file_name in os.listdir('captures'):
            file = "captures/" + file_name
            if os.path.isfile(file):
                os.remove(file)
    else:
        os.mkdir('captures')


def save_image(image, num_captures):
    filename = 'captures/capture' + str(num_captures) + '.png'
    log('saving image as "' + filename + '"')
    cv2.imwrite(filename, image)


if __name__ == '__main__':
    log('started')

    listener = Listener(address, authkey=b'password')

    init_directory()
    cam = VideoCapture(0)

    conn = listener.accept()
    calibrator_client = Client(calibrator.address, authkey=b'password')
    log('connected to calibrator')

    num_captures = 0
    try:
        while True:
            msg = conn.recv()
            if msg == 'capture':
                log('received capture message')

                result, image = cam.read()
                if result:
                    save_image(image, num_captures)
                    num_captures += 1

                    calibrator_client.send(image)
                else:
                    raise RuntimeError('Failed to capture image')
            elif msg == 'exit':
                log('received exit message')
                break
            else:
                log('error')
                raise ValueError('Unknown message "' + msg + '"')
    except Exception as e:
        log('error: ' + e.__class__.__name__)
        print(e)
    finally:
        listener.close()
        conn.close()
        calibrator_client.close()

        cam.release()