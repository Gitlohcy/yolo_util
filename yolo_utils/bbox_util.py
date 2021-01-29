from .general import *
import torch

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# def yolo2xyxy_2d(bboxes):
#     x_mid = bboxes[:, 0]
#     y_mid = bboxes[:, 1]
#     w = bboxes[:, 2]
#     h = bboxes[:, 3]
    
#     x1 = x_mid - (w/2)
#     y1 = y_mid - (h/2)
#     x2 = x_mid + (w/2)
#     y2 = y_mid + (h/2)

#     return np.array([x1, y1, x2, y2]).transpose(1,0)

def yolo2xyxy(bbox):
    x_mid,y_mid, w,h = bbox
    
    x1 = x_mid - (w/2)
    y1 = y_mid - (h/2)
    x2 = x_mid + (w/2)
    y2 = y_mid + (h/2)
    
    return [x1, y1, x2, y2]

# def xyxy2yolo_2d(bboxes):
#     x1 = bboxes[:, 0]
#     y1 = bboxes[:, 1]
#     x2 = bboxes[:, 2]
#     y2 = bboxes[:, 3]
    
#     w = x2 - x1
#     h = y2 - y1
#     x_mid = x1 + (w/2)
#     y_mid = x2 + (h/2)
    
#     return np.array([x_mid, y_mid, w, h]).transpose(1,0)


def xyxy2yolo(bbox):
    x1, y1, x2, y2 = bbox
    
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    x_mid = x1 + (w/2)
    y_mid = y1 + (h/2)
    
    return [x_mid, y_mid, w, h]