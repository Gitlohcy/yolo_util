
from .general import *

def images_from_l3_dir(pd_path: Path) -> dict:

    l3_dict = defaultdict(list)

    for metadata in ls(pd_path):
        for pd in ls(metadata):
            l3_dict[pd.name].append(ls(pd))

    l3_dict = {pd: flat(pd_list) for pd, pd_list in l3_dict.items()}
            
    return l3_dict

def images_from_l3_dir__top_1img(pd_path: Path, num_top_view: 1) -> dict:

    l3_dict = defaultdict(list)
    def get_product_list(product_name, angle):
        
        if angle == 'Top':
            return pd.Series(ls(pd_name)).sample(num_top_view)
        else:
            return ls(product_name)
    
    
    for angle_p in ls(pd_path):
        for pd_name in ls(angle_p):
            
            product_list = get_product_list(pd_name, angle_p.name)
            l3_dict[pd_name.name].append(product_list)
            

    l3_dict = {pd_name: flat(pd_list) for pd_name, pd_list in l3_dict.items()}
                
    return l3_dict

def text_list_to_df(label_list, new_class_id):
    
    for i,f in enumerate(label_list):
        file_name = f.name
        
        if i == 0:
            label_df = pd.read_csv(f, sep=' ', names=yolo_csv_column)
            label_df['file_name'] = file_name
        else:
            temp = pd.read_csv(f, sep=' ', names=yolo_csv_column)
            temp['file_name'] = file_name
            label_df = label_df.append(temp)
    
    label_df['class_id'] = new_class_id
    label_df.reset_index(drop=True, inplace=True)
    
        
    return label_df


def label_path_to_df(p: Path, new_class_id):
    return text_list_to_df(ls(p), new_class_id)

def bbox_row_with_filename(lines, txt: Path, until=None):
    #line[:-1] to remove extra space
    if until:
        return [[*line[:until].split(' ') , txt.name] for line in lines]
    return [[*line.split(' ') , txt.name] for line in lines]
