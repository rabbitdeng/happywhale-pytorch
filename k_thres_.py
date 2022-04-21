if __name__ == "__main__":
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
    import  faiss
    from sklearn.preprocessing import normalize
    import sklearn.metrics.pairwise as smp
    import seaborn as sns

    opt = parser.parse_args()

    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    import joblib

    N_SPLITS = 5
    MODEL_NAME = 'tf_efficientnet_b4_ns'
    NUM_CLASSES = 15587
    NO_NEURONS = 250
    EMBEDDING_SIZE = 512

    df_train = pd.read_csv("trainset.csv")
    df_valid = pd.read_csv("validset.csv")


    def k_t_search():
        best_map = 0.0
        best_thres = 0
        best_knn = 0
        for KNN in np.arange(100, 110, 10):
            train_embeddings = []
            trn_lbl_list = []

            with torch.no_grad():
                for data in tqdm(train_loader):
                    image, target = data
                    image, target = image.to(device), target.to(device)
                    embedding = model(image, labels=None)

                    train_embeddings.extend(embedding.detach().cpu().numpy())
                    trn_lbl_list.append(target.detach().cpu().numpy().tolist())

            image_embeddings = np.array(train_embeddings)
            trn_lbl_list = np.concatenate(trn_lbl_list)
            # from sklearn.manifold import TSNE
            # sns.set()
            # embeddings = TSNE(n_components=2, init='pca').fit_transform(image_embeddings)
            # ax = sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=trn_lbl_list, legend='full')
            # plt.plot(ax)
            # plt.show()
            #knn = NearestNeighbors(n_neighbors=KNN, metric="cosine")
            #knn = knn.fit(image_embeddings)
            val_lbl_list = []
            val_embeddings = []
            with torch.no_grad():
                for data in tqdm(valid_loader):
                    image, target = data
                    image, target = image.to(device), target.to(device)
                    embedding = model(image, labels=None)

                    val_embeddings.extend(embedding.detach().cpu().numpy())
                    val_lbl_list.append(target.detach().cpu().numpy().tolist())
            val_lbl_list = np.concatenate(val_lbl_list)
            val_image_embeddings = np.array(val_embeddings)

            train_emb = normalize( image_embeddings, axis=1, norm='l2')
            valid_emb = normalize(val_image_embeddings, axis=1, norm='l2')
            index = faiss.IndexFlatIP(EMBEDDING_SIZE)
            train_embeds = train_emb
            valid_embeds = valid_emb
            print(train_embeds.shape, valid_embeds.shape)
            index.add(train_embeds)
            dist, idx = index.search(valid_embeds, k=50)

            confs = 1 - dist
            best_acc = 0.0
            t = 0
            for thres in np.arange(0.1, 1.0, 0.1):
                sum_top5 = 0.0
                for i in range(len(valid_emb)):
                    index = idx[i][:5]
                    conf = confs[i][:5]
                    preds = trn_lbl_list[index]
                    if (conf < thres).any():
                        preds[-1] = -1
                    # 预测出与他相近的向量的index，表示其在embedding中的位置，也就是target_list
                    label = val_lbl_list[i]  # 第i个数据的标签

                    if label not in trn_lbl_list:
                        if preds[-1] == -1:
                            sum_top5 += 1.0
                        else:
                            pass
                    elif label in preds:
                        sum_top5 += 1.0
                    else:
                        pass
                map_ = sum_top5 / len(valid_emb)
                print(map_,thres)
                if map_ > best_acc:
                    best_acc = map_
                    t = thres
            print("validating map: %.5f ,thres = %.1f" % (best_acc, t))

        return best_map, best_knn, best_thres


    #kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=6)
    #df_train, df_valid = train_test_split(df, train_size=0.95, random_state=2333, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.manual_seed_all(seed=3407)
    # cross_validation
    PATIENCE = 2
    BEST_MAP5 = 0.0

    model = HappyWhaleModel(MODEL_NAME, NUM_CLASSES, NO_NEURONS, EMBEDDING_SIZE).to(device)
    # for fold_id, (train_idx, val_idx) in enumerate(kfold.split(df["image"], df["individual_key"])):
    train_data = WhaleDataset(df_train, "train")
    train_loader = DataLoader(train_data, batch_size=opt.batchSize, num_workers=2, shuffle=False)
    valid_data = WhaleDataset(df_valid, "valid")
    valid_loader = DataLoader(valid_data, batch_size=opt.batchSize, num_workers=2, shuffle=False)

    model.load_state_dict(torch.load("model_8_0.617947.pth")["net"])

    model.eval()
    cv_score, k, threshold = k_t_search()
    # print(f"on fold{fold_id},best cv score is {cv_score},k = {k},threshold = {threshold}")

    print(f"best cv score is {cv_score},k = {k},threshold = {threshold}")
