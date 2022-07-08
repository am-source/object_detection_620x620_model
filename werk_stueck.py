class WerkStueck:
    def __init__(self, box, score, color):

        # ymin, xmin, ymax, xmax = box
        self.bounding_box = box
        self.score = score
        self.color = color
