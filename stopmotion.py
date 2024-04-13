from losses import Focal_IoU
import argparse
import tensorflow as tf

import os
import cv2
import tqdm
import glob

import numpy as np

from cfg import IMAGE_HEIGHT, IMAGE_WIDTH

def inference(model, img):
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (IMAGE_WIDTH,IMAGE_HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x,verbose=0)

    if type(model.output) == list:
        num_of_output = len(model.output)
        return np.sum(pred,0) / num_of_output
    else: 
        return pred

def stopmotion(out_path, cap, mask, bg_image):
    frame_array  = []

    count = 0

    while True:
        success,image = cap.read()
        if success==False: 
            break

        reversed_mask = 1 - mask[count]
        cloth = image*mask[count] # 옷 추출
        bg = bg_image * reversed_mask # 옷제외한 부분 추출한
        result = cloth+bg

        frame_array.append(result)

    out = cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'DIVX'), 6, (IMAGE_WIDTH, IMAGE_HEIGHT))
    for count in range(len(frame_array)):
        if (count%5==0):
            out.write(np.uint8(frame_array[count]))
    out.release()

def main(model, video_path, bg_path):
    vidcap = cv2.VideoCapture(video_path)
    bg = cv2.imread(bg_path)
    fps = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    mask = np.zeros((fps, IMAGE_HEIGHT, IMAGE_WIDTH, 1),dtype=np.uint8)
    count = 0

    while True:
        success,image = vidcap.read()
        if success==False: 
            break
        pred = inference(model,image.copy())
        mask[count] =  (pred > 0.5).astype(np.uint8)
    
        if cv2.waitKey(10) == 27:                    
            break
        count += 1
    stopmotion(f"./result/{os.path.basename(video_path)}", vidcap, mask, bg)

if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type="str", default="")
    parser.add_argument("--video_path", type="str", default="")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.checkpoints, custom_objects = {'Focal_IoU':Focal_IoU})

    main(model, args.video_path)


