from .general import *


def yolo2xyxy_2d(bboxes):
    x_mid = bboxes[:, 0]
    y_mid = bboxes[:, 1]
    w = bboxes[:, 2]
    h = bboxes[:, 3]
    
    x1 = x_mid - (w/2)
    y1 = y_mid - (h/2)
    x2 = x_mid + (w/2)
    y2 = y_mid + (h/2)

    return np.array([x1, y1, x2, y2]).transpose(1,0)