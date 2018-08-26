import random
import os
import cv2
import pandas as pd

h, w = 345, 500

def gen_random_boxes(h=345, w=500, n_boxes=5):
    boxes = []
    for i in range(n_boxes):
        x_max = random.randint(75, w)
        y_max = random.randint(75, h)
        x_min = random.randint(10, x_max-10)
        y_min = random.randint(10, y_max-10)
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes

def read_images(directory):
    all_data=pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPG") or filename.endswith(".png"):
            img = cv2.imread(directory+filename)
            h, w, _ = img.shape
            boxes = gen_random_boxes(h=h,w=w)
            df=pd.DataFrame(boxes, columns=["x_min","y_min","x_max","y_max"])
            df["filename"]=filename
            df["label"]="true_neg"
            df["height"]=h
            df["width"]=w
            all_data = pd.concat([all_data,df])
            

    return all_data

if __name__ == '__main__':
    aug_data=read_images(DIR_PATH)
    aug_data.to_csv(FILFENAME+".csv",encoding="utf-8",index=False)
    
