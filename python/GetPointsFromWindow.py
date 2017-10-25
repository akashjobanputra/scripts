import cv2

__author__ = 'AkashJobanputra'

class GetPointsFromWindow:
    """
    Sets MouseCallback on LBUTTON_DOWN, provides function to get required points, or just line points
    """
    _multi_points = None
    _numberOfPoints = None

    def __init__(self, numberOfPoints, WindowName):
        """
        Initalize object
        :param numberOfPoints: Number of Points wanted
        :param WindowName: Window name in cv2.namedWindow
        """
        self._multi_points = []
        self._numberOfPoints = numberOfPoints
        cv2.setMouseCallback(WindowName, self._addPoints)

    def _addPoints(self, event, x, y, flags, param):
        """
        CallBack function for mouse event
        :param event:
        :param x:
        :param y:
        :param flags:
        :param param:
        :return:
        """
        if len(self._multi_points) == self._numberOfPoints:
            self._multi_points.clear()
        if event == cv2.EVENT_LBUTTONDOWN:
            self._multi_points.append((x,y))


    def getPoints(self):
        """
        Returns the points when required number of points are added in the list.
        :return: Empty list if all points are not received
        """
        if len(self._multi_points) == self._numberOfPoints:
            return self._multi_points
        return []

    def getLine(self):
        """
        Returns the last two coordinates
        :return: Empty list if mod 2 not equal to zero
        """
        if (len(self._multi_points) % 2) == 0:
            return self._multi_points[-2:]
        return []