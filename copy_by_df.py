
from .general import *
from shutil import copyfile

yolo_csv_column = ['class_id', 'x_center', 'y_center', 'width', 'height']


def save_file_by_row(row, dest, key_column=yolo_csv_column) :
    file_path = str(dest/row['file_name'])
    line = ' '.join(row[key_column].astype('str')) +'\n'

    with open(file_path, 'a+') as f:
        f.write(line)
        
'''copy file with pd.apply (full path required)'''
def copy_file_by_row(row, dest):
    file_name = Path(row['file_path']).name
    copyfile(str(row['file_path']), str(dest/file_name))

    
'''only copy image as the negative example'''
def negative_df_to_image(train_df, img_dest, sample_size=None, seed=123):
    if sample_size:
        train_df_sample = train_df.sample(frac=sample_size, random_state=seed)
        train_df_sample.apply(lambda x :copy_file_by_row(x, img_dest/'train'), axis=1)
    else:
        train_df.apply(lambda x :copy_file_by_row(x, img_dest/'train'), axis=1)


def train_df_to_image_label(train_df, label_dest, img_dest,  sample_size=None, seed=123):
    #save label
    
    if sample_size:
        train_df_sample = train_df.sample(frac=sample_size, random_state=seed)
        train_df_sample.apply(lambda x:save_file_by_row(x, label_dest/'train'), axis=1)
        train_df_sample.apply(lambda x :copy_file_by_row(x, img_dest/'train'), axis=1)
    else:
        train_df.apply(lambda x:save_file_by_row(x, label_dest/'train'), axis=1)
        train_df.apply(lambda x :copy_file_by_row(x, img_dest/'train'), axis=1)
    
    
def val_df_to_image_label(val_df, label_dest, img_dest, sample_size=None, seed=123):
    #save label
    
    if sample_size:
        val_df_sample = val_df.sample(frac=sample_size, random_state=seed)
        val_df_sample.apply(lambda x:save_file_by_row(x, label_dest/'val'), axis=1)
        val_df_sample.apply(lambda x :copy_file_by_row(x, img_dest/'val'), axis=1)
    else:
        val_df.apply(lambda x:save_file_by_row(x, label_dest/'val'), axis=1)
        val_df.apply(lambda x :copy_file_by_row(x, img_dest/'val'), axis=1)

def sample_df_to_image_label(df, label_dest,  img_dest, val_split=0.2) :
    df_val = df.sample(frac=val_split)
    df_train = df[~df.index.isin(df_val.index)]
    
    #save label
    df_train.apply(lambda x:save_file_by_row(x, label_dest/'train'), axis=1)
    df_val.apply(lambda x:save_file_by_row(x, label_dest/'val'), axis=1)

    #copy image
    df_train.apply(lambda x :copy_file_by_row(x, img_dest/'train'), axis=1)
    df_val.apply(lambda x :copy_file_by_row(x, img_dest/'val'), axis=1)

def len_dest(train_or_val: str, images_dest, label_dest):
    
    if train_or_val == 'train':
        print(len(ls(label_dest/'train')))
        if len(ls(label_dest/'train')) == len(ls(images_dest/'train')):
            print('match')
        else:
            print(len(ls(label_dest/'train')))
            print(len(ls(images_dest/'train')))
    elif train_or_val == 'val':
        print(len(ls(label_dest/'val')))
        if len(ls(label_dest/'val')) == len(ls(images_dest/'val')):
            print('match')
        else:
            print(len(ls(label_dest/'val')))
            print(len(ls(images_dest/'val')))
    
def create_img_lbl_dest(images_dest, label_dest):
    mkdir_r(images_dest/'train')
    mkdir_r(images_dest/'val')

    mkdir_r(label_dest/'train')
    mkdir_r(label_dest/'val')

    
