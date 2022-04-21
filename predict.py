# Libraries
import os
import gc

import time
import random
import math
from scipy import spatial
from tqdm import tqdm
import warnings
import cv2
import pandas as pd
import numpy as np
from numpy import dot, sqrt
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from IPython.display import display_html
import joblib
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from model import CustomModel, HappyWhaleModel
import timm
from dataset import WhaleDataset
from config import parser
from sklearn.neighbors import NearestNeighbors

MODEL_NAME = 'efficientnet_b0'
NUM_CLASSES = 15587
NO_NEURONS = 250
EMBEDDING_SIZE = 128

opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_name = "best_model_4_15.569172.pth"
df = pd.read_csv("train_new.csv")

# Path to trained model parameters (i.e. weights and biases)
classif_model_path = f"./{pretrained_name}"

# Load the model and append learned params
model = HappyWhaleModel(MODEL_NAME, NUM_CLASSES, NO_NEURONS, EMBEDDING_SIZE).to(device)
m_ = torch.load(classif_model_path)
model.load_state_dict(m_["net"])

# Retrieve all embeddings for each image
all_embeddings = []


def le_(df):
    from sklearn.preprocessing import LabelEncoder

    # encode object string label into integer label mapping
    le_species = LabelEncoder()
    le_species.fit(df.species)
    # le_species = joblib.load('le_species.pkl')
    df.species = le_species.transform(df.species)
    joblib.dump(le_species, f'le_species.pkl')
    from sklearn.preprocessing import LabelEncoder

    # encode object string label into integer label mapping
    le_individual_id = LabelEncoder()
    le_individual_id.fit(df.individual_id)
    # le_individual_id = joblib.load('le_individual_id.pkl')
    df.individual_id = le_individual_id.transform(df.individual_id)

    joblib.dump(le_individual_id, f'le_individual_id.pkl')
    return df.species, df.individual_id


df.species, df.individual_id = le_(df)

train_data = WhaleDataset(df, "train")
train_loader = DataLoader(train_data, batch_size=16, num_workers=0, shuffle=False)

model.eval()
with torch.no_grad():
    for data in tqdm(train_loader):
        image, target = data
        image, target = image.to(device), target.to(device)
        _, embedding = model(image, target)
        embedding = embedding.detach().cpu().numpy()
        all_embeddings.append(embedding)

# Concatenate batches together
image_embeddings = np.concatenate(all_embeddings)

# Save embeddings and corresponding image
np.save(f'{pretrained_name}.npy', image_embeddings)

knn = NearestNeighbors(n_neighbors=5, radius=0.5)
knn.fit(image_embeddings)
distances, indices = knn.kneighbors(image_embeddings)

# === PREDICTION ===
# Create the grouped predictions based on distances & indices
predictions = {"images": [], "individual_id": []}

for i in tqdm(range(len(image_embeddings))):
    index = indices[i]

    preds = df.iloc[index]["individual_id"]

    predictions["images"].append(index)
    predictions["individual_id"].append(preds)

print("!")
