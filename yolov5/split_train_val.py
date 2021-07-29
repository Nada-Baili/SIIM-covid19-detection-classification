import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

if not os.path.exists("./yolov5/covid19/train"):
    os.mkdir("./yolov5/covid19/train")
if not os.path.exists("./yolov5/covid19/val"):
    os.mkdir("./yolov5/covid19/val")
if not os.path.exists("./yolov5/covid19/train/images"):
    os.mkdir("./yolov5/covid19/train/images")
if not os.path.exists("./yolov5/covid19/val/images"):
    os.mkdir("./yolov5/covid19/val/images")
if not os.path.exists("./yolov5/covid19/train/labels"):
    os.mkdir("./yolov5/covid19/train/labels")
if not os.path.exists("./yolov5/covid19/val/labels"):
    os.mkdir("./yolov5/covid19/val/labels")

train_df = pd.read_csv("./input/train_image_level.csv")
labels= []
for i, (_, row) in enumerate(train_df.iterrows()):
    if (row.label!="none 1 0 0 1 1"):
        labels.append("opacity")
    else:
        labels.append("none")
train_df["labels"] = labels

train_split, valid_split = train_test_split(train_df, test_size=0.3, random_state=42, stratify=train_df.labels)
train_split = train_split.reset_index(drop=True)
valid_split = valid_split.reset_index(drop=True)

train_files = []
for i in range(len(train_split)):
    train_files.append("./yolov5/covid19/images/{}.jpg".format(train_split.id[i].split("_")[0]))

print("EXTRACTING TRAIN FILES")
tk = tqdm(enumerate(train_files), total=len(train_files))
for _, p in tk:
    shutil.copyfile(p, p.replace("images", "train//images"))
    try:
        shutil.copyfile(p.replace("images", "labels").replace("jpg", "txt"), p.replace("images", "train//labels").replace("jpg", "txt"))
    except:
        continue

val_files = []
for i in range(len(valid_split)):
    val_files.append(r"./yolov5/covid19/images/{}.jpg".format(valid_split.id[i].split("_")[0]))

print("EXTRACTING VALIDATION FILES")
tk = tqdm(enumerate(val_files), total=len(val_files))
for _, p in tk:
    shutil.copyfile(p, p.replace("images", "val//images"))
    try:
        shutil.copyfile(p.replace("images", "labels").replace("jpg", "txt"), p.replace("images", "val//labels").replace("jpg", "txt"))
    except:
        continue
