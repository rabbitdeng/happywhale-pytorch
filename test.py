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

df_train = pd.read_csv("trainset.csv")
df_valid = pd.read_csv("validset.csv")
cnt = 0
for idx, key in enumerate(df_valid.individual_key):
    flag = 0
    for i,item in enumerate (df_train.individual_key):
        if key ==item:
           flag = 1
           break
    if flag != 1:
        cnt += 1
        print("one")
        flag = 0

print(f"new individual : {cnt},{cnt / len(df_valid)}")
