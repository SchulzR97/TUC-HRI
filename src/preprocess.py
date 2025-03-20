from ultralytics import YOLO
import os
import numpy as np
import cv2 as cv
from pathlib import Path
from glob import glob
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset
import argparse

def get_ai_mask(img):
    results = yolo(img, conf=0.5, iou=0.45, verbose=False)
    segmentation_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    if results and results[0].masks:
        masks = results[0].masks.data.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        for class_id, mask in zip(class_ids, masks):
            if class_id != 0:
                continue
            mask = cv.resize(mask, (img.shape[1], img.shape[0]))
            segmentation_mask = np.logical_or(segmentation_mask, mask)

    segmentation_mask = segmentation_mask == False
    return segmentation_mask

def get_hsv_mask(img):
    hmin, hmax, smin, smax, vmin, vmax = 69, 87, 139, 255, 52, 255
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = (hmin, smin, vmin)
    upper = (hmax, smax, vmax)
    mask = cv.inRange(hsv, lower, upper)
    return mask > 0

def get_random_background(ds_backgrounds):
    record = ds_backgrounds[np.random.randint(0, len(ds_backgrounds))]
    bg = np.array(record['image'])
    bg = cv.cvtColor(bg, cv.COLOR_BGR2RGB)

    f = np.max([375 / bg.shape[0], 500 / bg.shape[1]])
    bg = cv.resize(bg, (0, 0), fx=f, fy=f)

    if bg.shape[0] > 375:
        sy = (bg.shape[0] - 375) // 2
        ey = sy + 375
        bg = bg[sy:ey, :, :]
    if bg.shape[1] > 500:
        sx = (bg.shape[1] - 500) // 2
        ex = sx + 500
        bg = bg[:, sx:ex, :]

    return bg

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_directory', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--cap_mode', type=str, default='realsense', help='Capture mode')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    in_dir = f'{args.dataset_directory}/sequences/{args.cap_mode}'
    out_dir = f'{args.out_dir}/sequences/{args.cap_mode}'

    ds_backgrounds = load_dataset('SchulzR97/backgrounds', split='train')

    yolo = YOLO('yolo11x-seg.pt')

    color_images = sorted(glob(f'{in_dir}/sequences/{args.cap_mode}/**/**/*_color.jpg'))

    prog = tqdm(color_images)
    last_seq_name = None
    for in_file in prog:
        in_file = Path(in_file)
        prog.set_description(f'{in_file.parent.parent.name}/{in_file.parent.name}/{in_file.name}')

        if in_file.parent.name != last_seq_name:
            bg = get_random_background(ds_backgrounds)

        in_file_depth = in_file.parent.joinpath(in_file.name.replace('_color', '_depth'))

        out_file = Path(out_dir).joinpath('sequences', args.cap_mode, in_file.parent.parent.name, in_file.parent.name, in_file.name)
        out_file_depth = out_file.parent.joinpath(out_file.name.replace('_color', '_depth'))
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            continue

        img = cv.imread(str(in_file))
        img_depth = cv.imread(str(in_file_depth), cv.IMREAD_UNCHANGED)
        
        hsv_mask = get_hsv_mask(img)
        ai_mask = get_ai_mask(img)
        combined_mask = np.logical_or(hsv_mask, ai_mask)

        phase = in_file.parent.parent.name

        #img_depth[combined_mask] = 13

        if phase == 'train':
            img[combined_mask, 0] = 0
            img[combined_mask, 1] = 255
            img[combined_mask, 2] = 0
        elif phase in ['val', 'test']:
            img[combined_mask, :] = bg[combined_mask, :]

        cv.imwrite(str(out_file), img)
        cv.imwrite(str(out_file_depth), img_depth)
        # cv.imshow('img', img)
        # cv.imshow('img_depth', img_depth)
        # cv.waitKey()

        last_seq_name = in_file.parent.name
        pass
    pass