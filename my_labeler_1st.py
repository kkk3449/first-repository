import os
from os.path import join
from glob import glob
import cv2
import numpy as numpy
import argparse
import numpy as np
import json
from pprint import pprint

args = argparse.ArgumentParser()

# hyperparameters
args.add_argument('img_path', type=str, nargs='?', default=None)
args.add_argument('mask_path', type=str, nargs='?', default=None)

config = args.parse_args()


def blend_mask(img_orig, img_mask, alpha=0.3):
    '''
    alpha : alpha blending ratio. 0 ~ 1
    '''
    imgBlack = np.zeros(img_mask.shape, dtype=np.uint8)
    mask = (img_mask / img_mask.max()) * 255
    mask = mask.astype(np.uint8)

    if len(np.unique(mask)) > 2:
        # multi channel mask
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        mask_white = cv2.merge((mask,mask,mask))
        mask_color = np.where(mask_white != 0, mask_color, 0)
    else:
        # 1 channel mask
        mask_color = cv2.merge((imgBlack, mask, mask))

    img_show = cv2.addWeighted(img_orig, 0.9, mask_color, alpha, 0.0)
    return img_show


def check_dir():
    flg_mask = True
    if config.mask_path is None \
        or len(config.mask_path) == 0 \
        or config.mask_path == '':
        print ('[*] mask file not exist')
        flg_mask = False
    
    if config.img_path is None \
        or len(config.img_path) == 0 \
        or config.img_path == '' \
        or os.path.isdir(config.img_path):
        root = os.path.realpath('./')
        if os.path.isdir(config.img_path):
            root = os.path.realpath(config.img_path)
        img_list = sorted(glob(join(root, '*.png')))
        img_list.extend(sorted(glob(join(root, '*.jpg'))))
        config.img_path = img_list[0]

    img_dir = os.path.dirname(os.path.realpath(config.img_path))
    mask_dir = os.path.dirname(os.path.realpath(config.mask_path)) if flg_mask else None
    mask_dir = os.path.realpath(config.mask_path) if flg_mask and os.path.isdir(config.mask_path) else mask_dir
    
    return img_dir, mask_dir, flg_mask


def move(pos, idx, img_list):
    if pos == 1:
        idx += 1
        if idx == len(img_list):
            idx = 0
    elif pos == -1:
        idx -= 1
        if idx == -1:
            idx = len(img_list) - 1
    return idx


def blend_view():
    cv2.namedWindow('show', 0)
    cv2.resizeWindow('show', 500, 500)

    img_dir, mask_dir, flg_mask = check_dir()

    fname, ext = os.path.splitext(config.img_path)
    img_list = [os.path.basename(x) for x in sorted(glob(join(img_dir,'*%s'%ext)))]
    
    dict_label = {}
    dict_label['img_dir'] = img_dir
    dict_label['mask_dir'] = img_dir
    dict_label['labels'] = []

    json_path = os.getenv('HOME')+'/aiffel/coarse_to_fine/annotation.json'
    json_file = open(json_path, 'w', encoding='utf-8')

    idx = img_list.index(os.path.basename(config.img_path))
    while True:
        start = cv2.getTickCount()
        fname = img_list[idx]
        mname = fname
        orig = cv2.imread(join(img_dir, fname), 1)

        img_show = orig
        if flg_mask:
            mask = cv2.imread(join(mask_dir, mname), 0) 
            img_show = blend_mask(orig, mask)
        
        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        
        print (f'[INFO] ({idx+1}/{len(img_list)}) {fname}... time: {time:.3f}ms')
    
        cv2.imshow('show', img_show)
        
        key = cv2.waitKey(0)
        if key == 27:   # Esc to Stop and Save Json result.
            return -1
        if key == ord('n'):
            idx = move(1, idx, img_list)
        elif key == ord('p'):
            idx = move(-1, idx, img_list)
        elif key == ord('f'):
            dict_label['labels'].append({'name':fname, 'class':1})
            idx = move(1, idx, img_list)
            print (f'[INFO] {fname}, class: true')
        elif key == ord('d'):
            dict_label['labels'].append({'name':fname, 'class':0})
            idx = move(1, idx, img_list)
            print (f'[INFO] {fname}, class: False')
        elif key == ord('v'):
            print ()
            pprint (dict_label)
            print ()
        elif key == ord('s'):
            json.dump(dict_label, json_file, indent=2)
            print (f'[INFO] < {json_path} > saved!')
    json_file.close()

if __name__ == '__main__':
    blend_view()
