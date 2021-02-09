from .general import *

import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import imgaug.augmenters as iaa
import boxx


def bgr2rgb_(img):
    return img[:, :, ::-1].transpose(2, 0, 1)

def rgb2bgr_(img):
    return img[:, :, ::-1].transpose(2, 0, 1)





#--bboxes--
def get_bbs(img_shape, labels: np.ndarray, has_cls_id=False):
    '''
    labels = [[cls_id, x1, y1, x2, y2]]
    '''
    if has_cls_id:
        bbs = [BoundingBox(*bbox, label=cls_id) for cls_id, *bbox in labels]
    else:
        bbs = [BoundingBox(*bbox) for bbox in labels]
    return BoundingBoxesOnImage(bbs, shape=img_shape)

def draw_bb(img, bbs, size=3):
    return bbs.draw_on_image(img, size=size)
    
def show_bb(img, bbs, size=3):
    ia.imshow(draw_bb(img, bbs, size))


#--segmentatinon mask--
def get_segmap(img, bool_mask):
    return SegmentationMapsOnImage(bool_mask, shape=img.shape)

def save_mask(mask, dest, fname):
    fname = dest/(Path(fname).stem+ '.npy')
    with open(str(fname), 'wb') as f:
        np.save(f, mask)
        
def read_mask(fpath):
    with open(str(fpath), 'rb') as f:
        mask = np.load(f)
    return mask

def draw_mask(img, mask):
    return mask.draw_on_image(img)[0]
    
def show_mask(img, mask):
    ia.imshow(draw_mask(img, mask))

