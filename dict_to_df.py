from .general import *

def bbox_row_with_filename(lines, txt: Path, until=None):
    #line[:-1] to remove extra space
    if until:
        return [[*line[:until].split(' ') , txt.name] for line in lines]
    return [[*line.split(' ') , txt.name] for line in lines]


def label_list_toDf(label_list: List[Path], column_name: list, class_name:str, product_id_dict):
    all_lines = []
    
    for txt in label_list:
        lines = open(str(txt)).read().splitlines()
        all_lines.append(bbox_row_with_filename(lines, txt, until=-1))    

    df = pd.DataFrame(flat(all_lines), columns=column_name)
    df['class_id'] = product_id_dict.get(class_name)
    df.reset_index(drop=True, inplace=True)
    
    df['file_stem'] = df.apply(lambda x: Path(x['file_name']).stem, axis=1)
    
    return df

def image_dict_to_df(img_dict):

    images_df = {k: pd.DataFrame(d, columns=['file_path']) for k, d in img_dict.items()}
    for k, d in img_dict.items():
        images_df[k]['file_stem'] = images_df[k].apply(lambda p: Path(p['file_path']).stem, axis=1)
    
    return images_df
    
def join_img_and_lbl_df(img_df, lbl_df):
    return {k: pd.merge(df, img_df[k], on='file_stem') for k,df in lbl_df.items()}
