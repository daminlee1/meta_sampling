class DetectionBox(object):
    def __init__(self, detection_class, xmin, ymin, xmax, ymax, score):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.width = xmax - xmin + 1
        self.height = ymax - ymin + 1
        self.area = self.width * self.height
        self.center_x = (self.xmin + self.xmax + 1)/2
        self.center_y = (self.ymin + self.ymax + 1)/2
        self.detection_class = detection_class
        self.score = score

