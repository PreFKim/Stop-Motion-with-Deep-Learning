import cv2
import tqdm
import glob
import os
import argparse

def convert(video_path, save_path, for_label=True):
    if for_label:
        save_path = save_path+"(label)"
        
    for filename in tqdm.tqdm(glob.glob(os.path.join(video_path,'*'))):
        vidcap = cv2.VideoCapture(filename)
        success,image = vidcap.read()

        count = 1
        success = True

        while 1:
            success,image = vidcap.read()
            if (success):
                print(image.shape)
            else:
                break
            image = cv2.resize(image, (960,640))
            
            if for_label:
                image = cv2.add(image, (-30, -30, -30,0))
                image[image<0] = 0
            
            cv2.imwrite(os.path.join(save_path, f"{os.path.basename(filename)}-{count:03d}.png"),image)
            print("saved image %d.png" % count)
            
            if cv2.waitKey(10) == 27: 
                break
            count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type="str", default="./data/videos")
    parser.add_argument("--save_path", type="str", default="./data/images")
    parser.add_argument("--for_label", action="store_true")

    args = parser.parse_args()

    convert(args.video_path, args.save_path, args.for_label)