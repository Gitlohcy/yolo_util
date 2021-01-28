import argparse
from .general import *
from .bbox_util import *
import .imgaug_util as iu
import .plot_util as pu
import .file_util as fu

import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.augmenters as iaa
import boxx

import torch
from torch import tensor
import uuid
from pycocotools.coco import COCO


#img aug augmentations
aug_seq1 = iaa.Sequential([
        iaa.Rotate((-180, 180), fit_output=True),
#         iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side": 480}),
#         iaa.Resize(0.5)
    ])
aug_back_img = iaa.Sequential([
    iaa.Resize({"shorter-side": "keep-aspect-ratio", "longer-side":1080}),
    ])

def aug_with_mask(img, mask, aug_seq):
    aug_img, aug_mask = aug_seq(image=img, segmentation_maps=mask)
    return aug_img, aug_mask


#get img, mask, cls_id
def get_rand_img_meta(coco: COCO, imgIds: List[int], n: int):
    chosen_img = list(pd.Series(imgIds).sample(n))
    return coco.loadImgs(chosen_img)

def img_mask_cls_id__from_coco_obj(img_dir: Path, coco: COCO, n_img: int, cname_map_dir: dict, filter_cls_name: List=None):

    catIds = coco.getCatIds(catNms=filter_cls_name) if filter_cls_name else coco.getCatIds()

    #get img_meta
    imgIds = flat([coco.getImgIds(catIds=[cat_id]) for cat_id in catIds])
    img_meta = get_rand_img_meta(coco, imgIds, 10)
    img_meta_dict = {meta['id']: meta for meta in img_meta}
    
    #get annotation
    annIds = coco.getAnnIds(imgIds=img_meta_dict.keys(), catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    #get img and mask
    img_n_masks = []
    for ann in anns:

        #find cls id
        cls_id = ann['category_id']
        c_name = coco.cats[cls_id]['name']
        
        img_id = ann['image_id']
        product_name = cname_map_dir.get(c_name)
        img_path = img_dir/product_name/img_meta_dict[img_id]['file_name']
        img = imageio.imread(img_path)
        
        mask = (coco.annToMask(ann) > 0)
        segmap = iu.get_seg_map(img, mask)

        img_n_masks.append([img, segmap, cls_id])
        
    return img_n_masks

def bbox_from_mask(bool_mask):
    mask_y, mask_x = np.array(np.where(bool_mask))
    x1 = np.min(mask_x)
    y1 = np.min(mask_y)
    x2 = np.max(mask_x)
    y2 = np.max(mask_y)

    return [x1, y1, x2, y2]

def bbox_from_pasted_mask(small_img_bool_mask, big_img_paste_coord):

    #tlbr on small img mask
    s_x1, s_y1, s_x2, s_y2 = bbox_from_mask(small_img_bool_mask)
    
    #coordinate of where small img mask paste on big img 
    b_x1, b_y1 = big_img_paste_coord #top left xy
    
    #tlbr of big img mask (pasted with small img mask)
    b_mask_x1 = b_x1 + s_x1
    b_mask_y1 = b_y1 + s_y1
    b_mask_x2 = b_x1 + s_x2
    b_mask_y2 = b_y1 + s_y2
    
    return [b_mask_x1, b_mask_y1,b_mask_x2, b_mask_y2]


#get random num
def random_xy(small_img, big_img):
    sih, siw = small_img.shape[:2]
    bih, biw = big_img.shape[:2]
    
    dh = abs(bih - sih)
    dw = abs(biw -siw)
    
    return np.random.randint(dw), np.random.randint(dh)

def rand_neg(limit):
    '''random with range negative limit until limit
    Example: 
    ->[np.random.randint(10) - 5 for _ in range(10)]
    ->[2, 3, -4, -1, -5, 1, -1, -1, 2, -4]
    '''
    return np.random.randint(limit*2) - limit

def rand_neg_with_positive_higher_chance(limit, chance_increase=0.2):
    return rand_neg(limit) + (limit * chance_increase)


#compute iou
def arr_clip_min(arr, min_a):
    return np.minimum(arr, np.repeat(min_a, len(arr)))

def np_iou_bbox(bbox1, bbox2, eps=1e-9):
    '''
    https://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    http://jsfiddle.net/Lqh3mjr5/
    
    bbox: rank 1 tensor
    '''
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1 
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2
    
    # Intersection area 
    x_overlap = np.min([b1_x2, b2_x2]) - np.max([b1_x1, b2_x1]) #leftmost x2 - rightmost x1
    y_overlap = np.min([b1_y2, b2_y2]) - np.max([b1_y1, b2_y1]) #topmost y2 - bottommost y1
    x_overlap, y_overlap = np.clip([x_overlap, y_overlap], a_min=0, a_max=None)
    inter =  x_overlap *  y_overlap
    
    #if is_overlap:
    if inter > 0:     
        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
    
        return inter / union #iou
    else:
        return 0
    
def torch_iou_bbox(bbox1, bbox2, eps=1e-9):
    '''
    https://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    http://jsfiddle.net/Lqh3mjr5/
    
    bbox: rank 1 tensor
    '''
    b1_x1, b1_y1, b1_x2, b1_y2 = tensor(bbox1) 
    b2_x1, b2_y1, b2_x2, b2_y2 = tensor(bbox2) 
    
    # Intersection area 
    x_overlap = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) #leftmost x2 - rightmost x1
    y_overlap = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0) #topmost y2 - bottommost y1
    inter =  x_overlap *  y_overlap
    
    #if is_overlap:
    if inter > 0:
        
        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
    
        return inter / union #iou
    
    else:
        return 0

def remove_overlap_bbox(bboxes, threshold=0.7):
    '''check overlap percentage  with each others
    and remove bbox if overlap percentage > threshold
    '''
    bboxes = np.array(bboxes)
    not_i = np.ones(len(bboxes), dtype='bool')
    is_label = np.ones(len(bboxes), dtype='bool')

    for i in range(len(bboxes)):
        not_i[i] = False
        for bbox2 in bboxes[not_i & is_label] :

            if np_iou_bbox(bboxes[i], bbox2) > threshold: 
                is_label[i] = False
                break
        not_i[i] = True

    return is_label


#paste by grid
def get_mid_of_each_cells(back_img_shape):
    bih, biw, _ = back_img_shape
    cells_wh = (biw//2, bih//2)
    cells_top_left = [0,0]

    #find mid point for each cells

    cells_mid_xys = []
    cells_mid_xys.append([cells_top_left[0] + cells_wh[0], cells_top_left[1]])
    cells_mid_xys.append([cells_top_left[0] , cells_top_left[1] + cells_wh[1]])
    cells_top_left[0] += cells_wh[0]
    cells_mid_xys.append([cells_top_left[0], cells_top_left[1]])
    cells_top_left[1] += cells_wh[1]
    cells_mid_xys.append([cells_top_left[0], cells_top_left[1]])
    
    #eg: [[540, 0], [0, 540], [540, 0], [540, 540]]
            
    return cells_mid_xys, cells_wh

def get_rand_spot_in_quarter_grid(num_to_paste, big_img, max_small_imgs_hw):
    cell_num = 4
    pasteable_shape = (big_img.shape[0] - (max_small_imgs_hw[0] ), #h
                       big_img.shape[1] - (max_small_imgs_hw[1] ), #w
                       3) #chnnl
    
    cells_mid_xys, cells_wh = get_mid_of_each_cells(pasteable_shape) #pasteable_shape
    cell_w, cell_h = cells_wh
    
    n_round = (num_to_paste // cell_num) + 1
    remainder = num_to_paste % cell_num
    
    paste_xys = []
    for i in range(n_round):
        
        #if last round randomly choose mid_xy with N remainder
        #else choose all
        n_to_pick = cell_num+1 if i != (n_round-1) else remainder

        paste_x, paste_x = (0, 0)
        for mid_x, mid_y in cells_mid_xys[:n_to_pick]: 
            offset_x = rand_neg_with_positive_higher_chance(cell_w) # rand_neg(cell_w)
            offset_y = rand_neg_with_positive_higher_chance(cell_h) # rand_neg(cell_h)

            paste_x = mid_x + offset_x
            paste_y = mid_y + offset_y            
            paste_xys.append([
                np.clip(paste_x, a_min=0, a_max=pasteable_shape[1]), #clip to width
                np.clip(paste_y, a_min=0, a_max=pasteable_shape[0]) #clip to height
            ])

    return paste_xys

def crop_small_img_into_bb_size(small_img, small_img_bool_mask):
    s_x1, s_y1, s_x2, s_y2 = bbox_from_mask(small_img_bool_mask)
    return small_img[s_y1:s_y2, s_x1:s_x2, :], small_img_bool_mask[s_y1:s_y2, s_x1:s_x2]

def paste_into_image_(small_img, small_img_segmap, cls_id,  big_img, x1y1=None):
    #get random xyxy
    x1,y1 = x1y1 if x1y1 else random_xy(small_img, big_img)        
    x1, y1 = int(x1), int(y1)
    
    x2 = small_img.shape[1] + x1
    y2 = small_img.shape[0] + y1

    #get mask
    small_img_mask = small_img_segmap.get_arr()
    

    #solve rounding issue of small img_w
    dy = abs(y2-y1) - small_img_mask.shape[0]
    dx = abs(x2-x1) - small_img_mask.shape[1]
    y1 += dy
    x1 += dx
    x2 += dx
    y2 += dy
    
    #paste with mask and get bbox
    big_img[y1:y2, x1:x2, :][small_img_mask] = small_img[small_img_mask] #big_img[h,w,chnl]
    bbox = bbox_from_pasted_mask(small_img_mask, [x1,y1])
        
    return big_img, bbox, cls_id

def paste_mask_n_imgs_by_grid(big_img, img_masks_cls_id):
    
    num_to_paste = len(img_masks_cls_id)
    sih, siw, _ = [*zip(*[img.shape for img, mask, cls_id in img_masks_cls_id])]
    max_sih = np.max(sih)
    max_siw = np.max(siw)
    
    paste_xys = get_rand_spot_in_quarter_grid(num_to_paste, big_img, (max_sih, max_siw))

    
    i=0
    cls_bboxes = []
    for x1y1 in paste_xys:
        big_img, bbox, cls_id = paste_into_image_(*img_masks_cls_id[i], big_img, x1y1=x1y1)
        cls_bboxes.append([cls_id, *bbox])
        i += 1

    cls_bboxes = np.array(cls_bboxes)
    bboxes = cls_bboxes[:, 1:5]
    is_label = remove_overlap_bbox(bboxes, 0.6)
        
    return big_img, cls_bboxes[is_label]

def paste_mask_imgs_by_grid(big_img, small_imgs, small_img_segmaps):
    num_to_paste = len(small_imgs)
    bih, biw, _ = big_img.shape
    n_row, n_col = grid_shape

    ch = bih // n_col #cell height
    cw = biw // n_row  #cell width

    cell_top_left = []
    c_x, c_y = (0,0)
    for _ in range(3):
        c_x += ch
        c_y += cw
        cell_top_left.append([c_x, c_y])


    paste_xys = get_rand_spot_in_quarter_grid(num_to_paste, big_img)

    i=0
    bboxes = []
    for x1y1 in paste_xys:
        big_img, bbox = paste_into_image_(small_img[i], small_img_mask[i], big_img, x1y1=x1y1)
        bboxes.append(bbox)
        i += 1

    bboxes = remove_overlap_bbox(bboxes, 0.8)
                   
    return big_img, bboxes

# write labels
def uniq_file_name(prefix=None):
    hex_name = uuid.uuid4().hex
    uniq_name = prefix + '_' + hex_name if prefix else  hex_name
    return uniq_name

def list_2_yoloLabel_lines(img, yolo_bboxes):
    ih, iw , _ = img.shape
    yolo_bboxes = np.array(yolo_bboxes).astype('float')
    
    yolo_bboxes[:, 1:5]  = (yolo_bboxes[:, 1:5] / [iw, ih, iw, ih]).round(4)
    str_lines = yolo_bboxes.astype('str')
    return str_lines

def f_writelines(lines: List[str], fname, split_by=' '):
    lines = [split_by.join(list(line)) +'\n' for line in lines]
    
    with open(str(fname), 'w') as f:
        f.writelines(lines)

def write_img_and_bboxes(img, labels, img_dest, lbl_dest):
    fname = uniq_file_name('img')
    img_name = fname + '.jpg'
    lbl_name = fname + '.txt'
    
    imageio.imwrite(img_dest/img_name, img)
    yolo_lines = list_2_yoloLabel_lines(img, labels)
    f_writelines(yolo_lines, lbl_dest/lbl_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--back-img-dir', type=str, default=None, help='path to background image')
    parser.add_argument('--back-img-paths-txt', type=str, default=None, help='path to list of background image txt')
    parser.add_argument('--small-img-dir', type=str, default=None, help='path to image used for pasting')
    parser.add_argument('--coco-json', type=str, default=None, help='path to json file (small img labels)')
    parser.add_argument('--img-dest', type=str, default=Path.cwd()/'test_img_dir')
    parser.add_argument('--lbl-dest', type=str, default=Path.cwd()/'test_lbl_dir')
    parser.add_argument('--num2gen', type=int, default=10, help='num of image to generate')
    parser.add_argument('--paste-num', type=int, default=8, help='nom of image to paste for each image generation')
    parser.add_argument('--cls-map', type=str, help='yaml path to load cls_name mapping')
    
    opt = parser.parse_args()

    img_file_type = ['*.jpg', '*.png']
    small_img_dir = Path(opt.small_img_dir)
    img_dest = Path(opt.img_dest)
    lbl_dest = Path(opt.lbl_dest)
    coco_json = Path(opt.coco_json)

    # assert back_img_dir.is_dir(), 'back_img_dir is not a valid directory'
    assert coco_json.is_file() and coco_json.suffix == '.json', 'coco_json is not a valid json file'

    if not img_dest.is_dir():
        img_dest.mkdir()

    if not lbl_dest.is_dir():
        lbl_dest.mkdir()

    if opt.back_img_dir is not None:
        back_img_dir = Path(opt.back_img_dir)
        back_img_paths  = pd.Series(flat([back_img_dir.ls(img_type) for img_type in img_file_type]))
    else:
        back_img_paths_txt = Path(opt.back_img_paths_txt)
        assert(back_img_paths_txt.is_file(), 'invalid back img paths defined')
        back_img_paths = fu.f_readlines(back_img_paths_txt)
        

coco=COCO(str(coco_json))
# img_n_masks = img_mask_cls_id__from_coco_obj(s1_horizontal, coco, 10)


modified_back_imgs = []
cls_name_map_dir = load_yaml(opt.cls_map)

for i in range(opt.num2gen):
    img_n_masks = img_mask_cls_id__from_coco_obj(small_img_dir, coco, opt.paste_num, cls_name_map_dir)

    new_img_masks_cls_id = []
    for small_img, segmap, cls_id in img_n_masks:
        aug_img, aug_mask = aug_with_mask(small_img, segmap, aug_seq1)
        aug_img_mask = aug_mask.get_arr()
        small_img, small_img_mask = crop_small_img_into_bb_size(aug_img, aug_img_mask)
        new_seg_map = iu.get_seg_map(small_img, small_img_mask)
        new_img_masks_cls_id.append([small_img, new_seg_map, cls_id])

    #get back_img
    rand_id = np.random.randint(len(back_img_paths))
    back_img_p = back_img_paths[rand_id]

    back_img = imageio.imread(back_img_p)
    back_img = aug_back_img(image=back_img)
    
    back_img, bboxes = paste_mask_n_imgs_by_grid(back_img, new_img_masks_cls_id)
    
    write_img_and_bboxes(back_img, bboxes, img_dest, lbl_dest)

# cname = ['hundred_plus', 'cincau', 'mimi']
# img_n_masks = img_n_mask_from_coco_obj(s1_horizontal, coco_h, 10, filter_cls_name=cname)


