import numpy as np
import os
import cv2
import glob
import argparse


def get_mask(image_path, save_path):
    
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)

    for filename in glob.glob(os.path.join(image_path,"*")):
        img = cv2.imread(filename)
        img = cv2.inRange(img, [226,226,226], [255,255,255]) # 해당 범위 내의 픽셀은 
        img = img // 255
        cv2.imwrite(os.path.join(save_path,os.path.basename(filename)),img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="./data/images(label)")
    parser.add_argument("--mask_path", type=str, default="./data/masks")

    args = parser.parse_args()