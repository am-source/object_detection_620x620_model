import cv2


class MyCamera:
    def __init__(self, cam_source=0):
        self.camera = cv2.VideoCapture(cam_source)
        self.frame = None

    def frame_available(self):
        ret, frame = self.camera.read()
        if ret:
            self.frame = frame
        return ret

    def release(self):
        self.camera.release()
