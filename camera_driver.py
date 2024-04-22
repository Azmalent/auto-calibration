from cv2 import VideoCapture
from getopt import getopt
from multiprocessing.connection import Client, Listener
import cv2
import sys

class CameraDriver():
    def __init__(self):
        self.cam = VideoCapture(0)

    
    def log(self, message):
        print('[Camera Driver] ' + message)


    def capture(self):
        result, image = self.cam.read()
        return result, image


if __name__ == '__main__':
    listener_port = None
    client_port = None

    opts, args = getopt(sys.argv[1:], '', ['listener-port=', 'client-port='])
    for opt, arg in opts:
        if opt == '--listener-port':
            listener_port = int(arg)
        elif opt == '--client-port':
            client_port = int(arg)

    listener = Listener(('localhost', listener_port), authkey=b'password')

    driver = CameraDriver()

    conn = listener.accept()
    calibrator_client = Client(('localhost', client_port), authkey=b'password')
    
    try:
        while True:
            msg = conn.recv()
            if msg == 'capture':
                driver.log('received capture message')

                result = False
                while not result:
                    result, image = driver.capture()
                    if result:
                        calibrator_client.send(image)
            elif msg == 'exit':
                driver.log('received exit message')
                break
            else:
                driver.log('error')
                raise ValueError('Unknown message "' + msg + '"')
    except Exception as e:
        driver.log('error: ' + e.__class__.__name__)
        print(e)
    finally:
        listener.close()
        conn.close()
        calibrator_client.close()

        driver.cam.release()