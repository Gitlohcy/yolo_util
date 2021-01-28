import argparse
import pyperclip


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', type=str, default='general')
    opt = parser.parse_args()



    import_general ="""import sys
    from os.path import expanduser
    home = expanduser("~")
    sys.path.append(home+'/Documents/GitHub/python-util')
    """

    import_imgaug ="""
    import imageio
    import imgaug as ia
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

    import imgaug.augmenters as iaa
    import boxx
    """

    lib_dict = {
        'general': import_general,
        'imgaug': import_imgaug,
    }


    cp_str = lib_dict.get(opt.lib)
    pyperclip.copy(cp_str)

    # def pycopy(to_cp):
    #     pyperclip.copy(to_cp)



