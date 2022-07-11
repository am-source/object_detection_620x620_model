import cv2


class MyCamera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.frame = None

    def frame_available(self):
        ret, frame = self.camera.read()
        if ret:
            self.frame = frame
        return ret
