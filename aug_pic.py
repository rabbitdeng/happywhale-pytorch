import pandas as pd
import os
# use for generate new washed csv file of current train images left.
import torchvision.transforms.transforms

from dataset import transform
import PIL.Image as Image
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

pics_dir = "D:/hw_ptlight/data/convert-backfintfrecords/happy-whale-and-dolphin-backfin/train_images"

df_train = pd.read_csv(
    "D:/hw_ptlight/output/working//train_encoded_folded.csv")


def count_sample(df):
    count = torch.zeros(15587)
    for _, i in enumerate(df.individual_key):
        count[i] += 1
        # print(_,count[i])

    result = count
    print(result)
    return result


df_new = pd.DataFrame()
# list_existpic = os.listdir("augedimages")
for idx, k in enumerate(df_train.kfold):
    print(idx)
    if k != 0.0:
        print("transforming")
        img = df_train.iloc[idx].image
        # if img in list_existpic
        pic = Image.open(os.path.join(pics_dir, img)).convert("RGB")
        pic = np.array(pic)
        pic_new = transform["train"](image=pic)
        pic_new = pic_new["image"]
        # tensor_to_pil = torchvision.transforms.transforms.ToPILImage()
        pic_new = Image.fromarray(pic_new).convert("RGB")
        pic_new.save(
            f"D:/hw_ptlight/data/convert-backfintfrecords/happy-whale-and-dolphin-backfin/auged_train_images/A{img}")
        new_row = df_train.iloc[idx]
        new_row.image = f"A{img}"
        new_row.image_path = f"data\\convert-backfintfrecords\\happy-whale-and-dolphin-backfin\\auged_train_images\\A{img}"
        df_new = df_new.append(new_row)

df_train = df_train.append(df_new, ignore_index=True)

print("finish")
df_train.to_csv( "D:/hw_ptlight/output/working/train_auged_encoded_folded.csv",
                index=False)
