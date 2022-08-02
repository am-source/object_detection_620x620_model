import cv2


class MyCamera:
    def __init__(self, cam_source=0, res_width=1920, res_height=1080):
        self.camera = cv2.VideoCapture(cam_source)
        self.frame = None
        self.set_res(res_width, res_height)  # For 1080P

    def set_res(self, width, height):
        self.camera.set(3, width)
        self.camera.set(4, height)

    def frame_available(self):
        ret, frame = self.camera.read()
        if ret:
            self.frame = frame
        return ret

    def release(self):
        self.camera.release()
