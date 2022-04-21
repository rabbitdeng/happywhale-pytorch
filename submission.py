if __name__ == "__main__":
    import os
    import gc
    import pandas as pd
    import torch
    from torchvision.transforms import transforms
    from model import CustomModel, HappyWhaleModel
    from dataset import WhaleDataset

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

    opt = parser.parse_args()

    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    import joblib

    N_SPLITS = 5
    MODEL_NAME = 'efficientnet_b0'
    NUM_CLASSES = 15587
    NO_NEURONS = 250
    EMBEDDING_SIZE = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.manual_seed_all(seed=3407)
    # cross_validation
    df_train = pd.read_csv("train_encoded.csv")
    df_test = pd.read_csv("sample_submission.csv")
    model = HappyWhaleModel(MODEL_NAME, NUM_CLASSES, NO_NEURONS, EMBEDDING_SIZE).to(device)
    train_data = WhaleDataset(df_train, "train")
    train_loader = DataLoader(train_data, batch_size=32, num_workers=2, shuffle=False)
    test_data = WhaleDataset(df_test, "test")
    test_loader = DataLoader(test_data, batch_size=32, num_workers=2, shuffle=False)

    model.load_state_dict(torch.load("fold_0model_11_9.197008.pth")["net"])

    model.eval()
    k = 100
    thres = 0.6
    knn = NearestNeighbors(n_neighbors=k, metric="cosine")

    train_embeddings = []
    trn_lbl_list = []

    threshold = thres

    with torch.no_grad():
        for data in tqdm(train_loader):
            image, target = data
            image, target = image.to(device), target.to(device)
            _, embedding = model(image, target)

            train_embeddings.extend(embedding.detach().cpu().numpy())
            trn_lbl_list.append(target.detach().cpu().numpy().tolist())

    image_embeddings = np.array(train_embeddings)
    trn_lbl_list = np.concatenate(trn_lbl_list)

    knn.fit(image_embeddings)
    test_embeddings = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            image = data
            image = image.to(device)
            embedding = model(image)

            test_embeddings.extend(embedding.detach().cpu().numpy())

    test_image_embeddings = np.array(test_embeddings)

    distances, indices = knn.kneighbors(test_image_embeddings)
    confs = 1 - distances
    # === PREDICTION ===

    sum_top5 = 0.0
    pbar = tqdm(range(len(test_image_embeddings)))
    map_ = 0.0
    predictions = []
    preds_decoded = {}
    for i in pbar:
        index = indices[i][:5]
        conf = confs[i][:5]
        # preds = np.array(trn_lbl_list)[index]
        preds = df_train.iloc[index]["individual_id"].values
        if conf[0] < threshold:
            templist = ['new_individual', preds[1], preds[2], preds[3], preds[4]]
        elif conf[1] < threshold:
            templist = [preds[0], 'new_individual', preds[2], preds[3], preds[4]]
        elif conf[2] < threshold:
            templist = [preds[0], preds[1], 'new_individual', preds[3], preds[4]]
        elif conf[3] < threshold:
            templist = [preds[0], preds[1], preds[2], 'new_individual', preds[4]]
        elif conf[4] < threshold:
            templist = [preds[0], preds[1], preds[2], preds[3], 'new_individual']
        else:
            templist = preds

        preds_decoded[df_test.iloc[i]["image"]] = templist

    for x in tqdm(preds_decoded):
        preds_decoded[x] = ' '.join(preds_decoded[x])

    predictions = pd.Series(preds_decoded).reset_index()
    predictions.columns = ['image', 'predictions']
    predictions.to_csv('submission.csv', index=False)
    predictions.head()
