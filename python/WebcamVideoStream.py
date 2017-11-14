import cv2
from threading import Thread
from uuid import getnode as get_mac


class WebcamVideoStream:
    def __init__(self, src, width=480, height=368):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.frame = None
        self.src = src
        if src == 'picam' and self.checkIfRaspberryPi():
            from picamera.array import PiRGBArray
            from picamera import PiCamera
            self.stream = PiCamera()
            self.stream.framerate = 32
            self.stream.resolution = (width, height)
            self.rawCapture = PiRGBArray(self.stream, size=(width, height))
        elif src != 'picam':
            self.stream = cv2.VideoCapture(src)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            (self.grabbed, self.frame) = self.stream.read()
        else:
            raise Exception('Not A Raspberry Pi Device')

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        self._stop = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        if self.src == 'picam' and self.checkIfRaspberryPi():
            sleep(0.5)
            for image in self.stream.capture_continuous(self.rawCapture, format='bgr', use_video_port=True):
                if self._stop:
                    # self.stream.close()
                    # print('stopped')
                    self.stopped = True
                    return
                self.frame = image.array
                self.frame.setflags(write=1)
                self.rawCapture.truncate()
                self.rawCapture.seek(0)
        elif self.src != 'picam':
            while True:
                # if the thread indicator variable is set, stop the thread
                if self._stop:
                    # self.stream.release()
                    self.stopped = True
                    return

                # otherwise, read the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()
        else:
            return

    def read(self):
        # return the frame most recently read
        while self.frame is None:
            pass
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self._stop = True

    def release(self):
        while not self.stopped:
            pass
        # print('release')
        if self.src == 'picam':
            self.stream.close()
        else:
            self.stream.release()

    def checkIfRaspberryPi(self):
        if hex(get_mac())[2:6] == 'b827':
            return True
        return False
