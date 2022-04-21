import os
import gc
import pandas as pd
import torch
from torchvision.transforms import transforms
from model import HappyWhaleModel
from dataset import WhaleDataset
from train import le_
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import parser
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import sklearn.metrics.pairwise as smp
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import joblib

if __name__ == "__main__":
    opt = parser.parse_args()

    df_train = pd.read_csv("train_encoded.csv")
    df_test = pd.read_csv("sample_submission.csv")
    train_data = WhaleDataset(df_train, "train")
    train_loader = DataLoader(train_data, batch_size=opt.batchSize, num_workers=0, shuffle=False)
    test_data = WhaleDataset(df_test, "test")
    test_loader = DataLoader(test_data, batch_size=opt.batchSize, num_workers=0,shuffle=False)

    knn = NearestNeighbors(n_neighbors=100, metric="cosine")

    train_images = []

    with torch.no_grad():
        for data in tqdm(train_loader):
            image, _ = data
            image = image.flatten(1)
            train_images.extend(image.detach().cpu().numpy())

    image_embeddings = np.array(train_images)

    knn.fit(image_embeddings)

    test_images = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            image = data
            image = image.flatten(1)
            test_images.extend(image.detach().cpu().numpy())

    distances, indices = knn.kneighbors(X=test_images)
    confs = 1 - distances
    nearest_df = pd.DataFrame()
    for idx in range(len(test_images)):
        index = indices[idx][:20]
        nearest_df = nearest_df.append(df_train.iloc[index])


    print(len(nearest_df))
    nearest_df = nearest_df.drop_duplicates()
    print(len(nearest_df))
    nearest_df.to_csv("nearest_trainset.csv",index=False)
