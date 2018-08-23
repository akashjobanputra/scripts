"""
    Basic helper class for managing video capture streams.

    TODO: Exception Handling if source already exits
"""

import cv2


class MultiVideoCapture:
    def __init__(self):
        self.streams = {}

    def getFrames(self):
        # TODO: add a stream parameter for getting frames from a particular stream
        for stream_key in self.streams:
            ret, frame = self.streams[stream_key]['stream'].read()
            yield ret, frame, stream_key

    def addStream(self, source, stream_id=None):
        if stream_id in self.streams:
            raise Exception('Stream ID Already Exists')
        if stream_id is None:
            stream_id = len(self.streams)
        camStream = cv2.VideoCapture(source)
        self.streams.update({stream_id: {'stream': camStream, 'source': source}})

    def __del__(self):
        # print('IN: __del__')
        for stream in self.streams.values():
            stream['stream'].release()


def main():
    testObj = MultiVideoCapture()
    testObj.addStream('rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov')
    testObj.addStream(0)
    for _ in range(10):
        for ret, _, s_id in testObj.getFrames():
            print(s_id, ret)


if __name__ == '__main__':
    main()
