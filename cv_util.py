import cv2


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
