import gc
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import torch

from model import HappyWhaleModel, softmax_loss, TripletLoss
from dataset import WhaleDataset

from tqdm import tqdm
from config import parser
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import  normalize
from sklearn.neighbors import NearestNeighbors


import faiss
opt = parser.parse_args()

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import joblib

N_SPLITS = 5
MODEL_NAME = 'convnext_base_384_in22ft1k'
NUM_CLASSES = 15587
NO_NEURONS = 250
EMBEDDING_SIZE = 512
gc.collect()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.manual_seed_all(seed=3407)

model = HappyWhaleModel(MODEL_NAME, NUM_CLASSES, NO_NEURONS, EMBEDDING_SIZE).to(device)

df_train = pd.read_csv("trainset.csv")
df_valid = pd.read_csv("validset.csv")

train_data = WhaleDataset(df_train, "train")
# train_sampler = WeightedRandomSampler(weights=weight_sample, num_samples=len(train_data), replacement=True)

train_loader = DataLoader(train_data, batch_size=opt.batchSize, num_workers=2, shuffle=True,
                          )
valid_data = WhaleDataset(df_valid, "valid")
valid_loader = DataLoader(valid_data, batch_size=opt.batchSize, num_workers=2, shuffle=False
                          )


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            select = (target != 0).type(torch.LongTensor).cuda()
            at = self.alpha.gather(0, select.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=6)


def accuracy(train_emb, train_list, valid_emb, valid_list):
    """Computes the accuracy """
    KNN = 100
    knn = NearestNeighbors(n_neighbors=KNN, metric="cosine")
    knn = knn.fit(train_emb)
    distances, indices = knn.kneighbors(X=valid_emb, n_neighbors=KNN)
    confs = 1 - distances
    best_acc = 0.0
    t = 0
    for thres in np.arange(0.1, 1.0, 0.1):
        sum_top5 = 0.0
        for i in range(len(valid_emb)):
            index = indices[i][:5]
            conf = confs[i][:5]
            preds = train_list[index]
            if (conf < thres).any():
                preds[-1] = -1
            # 预测出与他相近的向量的index，表示其在embedding中的位置，也就是target_list
            label = valid_list[i]  # 第i个数据的标签

            if label not in train_list:
                if preds[-1] == -1:
                    sum_top5 += 1.0
                else:
                    pass
            elif label in preds:
                sum_top5 += 1.0
            else:
                pass
            map_ = sum_top5 / len(valid_emb)

            if map_ > best_acc:
                best_acc = map_
                t = thres
    print("validating map: %.5f ,KNN = %d,thres = %.1f" % (best_acc, KNN, t))
    return best_acc


def acc_(train_emb, train_list, valid_emb, valid_list):
    train_emb = normalize(train_emb, axis=1, norm='l2')
    valid_emb = normalize(valid_emb, axis=1, norm='l2')
    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    train_embeds = train_emb
    valid_embeds = valid_emb
    print(train_embeds.shape, valid_embeds.shape)
    index.add(train_embeds)
    dist, idx = index.search(valid_embeds, k=100)

    confs = 1 - dist
    best_acc = 0.0
    t = 0
    for thres in np.arange(0.1, 1.0, 0.1):
        sum_top5 = 0.0
        for i in range(len(valid_emb)):
            index = idx[i][:5]
            conf = confs[i][:5]
            preds = train_list[index]
            if (conf < thres).any():
                preds[-1] = -1
            # 预测出与他相近的向量的index，表示其在embedding中的位置，也就是target_list
            label = valid_list[i]  # 第i个数据的标签

            if label not in train_list:
                if preds[-1] == -1:
                    sum_top5 += 1.0
                else:
                    pass
            elif label in preds:
                sum_top5 += 1.0
            else:
                pass
            map_ = sum_top5 / len(valid_emb)

            if map_ > best_acc:
                best_acc = map_
                t = thres
    print("validating map: %.5f ,thres = %.1f" % (best_acc, t))
    return best_acc

def one_epoch_acc():
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    t_emb = []
    t_list = []
    model.eval()
    acc = 0.0
    df_train = pd.read_csv("trainset.csv")
    train_data = WhaleDataset(df_train, "train_noaug")
    # train_sampler = WeightedRandomSampler(weights=weight_sample, num_samples=len(train_data), replacement=True)

    train_loader = DataLoader(train_data, batch_size=opt.batchSize, num_workers=2, shuffle=True,
                              )
    valid_data = WhaleDataset(df_valid, "valid")
    valid_loader = DataLoader(valid_data, batch_size=opt.batchSize, num_workers=2, shuffle=False
                              )
    with torch.no_grad():
        for i, data in enumerate(train_loader, 0):
            inputs, label = data
            inputs = inputs.to("cuda")
            label = label.to("cuda")

            arc_output, emb = model(inputs, label)
            t_emb.extend(emb.detach().cpu().numpy())
            t_list.append(label.detach().cpu().numpy().tolist())

        t_list = np.concatenate(t_list)
        t_emb = np.array(t_emb)
        v_emb = []
        v_list = []
        for i, data in enumerate(valid_loader, 0):
            inputs, label = data
            inputs = inputs.to("cuda")
            label = label.to("cuda")

            arc_output, emb = model(inputs, label)
            v_emb.extend(emb.detach().cpu().numpy())
            v_list.append(label.detach().cpu().numpy().tolist())

    v_list = np.concatenate(v_list)
    v_emb = np.array(v_emb)

    acc = acc_(train_emb=t_emb, train_list=t_list, valid_emb=v_emb, valid_list=v_list)
    return acc



def le_(df):
    from sklearn.preprocessing import LabelEncoder

    le_species = LabelEncoder()
    le_species.fit(df.species)

    df.species = le_species.transform(df.species)
    joblib.dump(le_species, f'le_species.pkl')
    from sklearn.preprocessing import LabelEncoder

    le_individual_id = LabelEncoder()
    le_individual_id.fit(df.individual_id)
    # le_individual_id = joblib.load('le_individual_id.pkl')
    individual_key = le_individual_id.transform(df.individual_id)

    joblib.dump(le_individual_id, f'le_individual_id.pkl')
    return individual_key


def trainwithvalid():
    PATIENCE = 7

    start_epoch = -1

    criterion_fl = FocalLoss(gamma=2)

    still = 0
    best_valid = 0.0

    lr_adam = 3e-4
    lr_sgd = 3e-5

    # for fold_id, (train_idx, val_idx) in enumerate(kfold.split(df["image"], df["individual_key"])):
    optimizer_adam = torch.optim.AdamW([
        {'params': model.fc.parameters(),
         'weight_decay': 1e-6,
         'lr': lr_adam},
        {'params': model.pooling.parameters(),
         'weight_decay': 1e-6,
         'lr': lr_adam},

        {'params': model.arc_head.parameters(),
         'weight_decay': 1e-6,
         'lr': lr_adam},
        {'params': model.model.parameters(), 'weight_decay': 1e-6,
         'lr': 3e-5},
    ])
    optimizer = torch.optim.SGD(params=model.parameters(), momentum=0.9,
                                lr=lr_sgd)
    lrschedule_adam = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer_adam, T_0=20,
                                                                           eta_min=1E-6,
                                                                           last_epoch=start_epoch)
    lrschedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6,
                                                            last_epoch=start_epoch)
    # skf = list((df.image.values,df.individual_key.values))
    model.load_state_dict(torch.load("model_13_0.720611.pth")["net"])
    optimizer_adam.load_state_dict(torch.load("model_13_0.720611.pth")["optimizer_adam"])
    start_epoch = torch.load("model_13_0.720611.pth")["epoch"]
    lrschedule_adam = torch.load("model_13_0.720611.pth")["lrschedule_adam"]
    for epoch in range(start_epoch + 1, opt.epochs):
        print(get_lr(optimizer_adam))
        running_loss = 0.0
        cnt = 0
        pbar = tqdm(train_loader)

        model.train()

        avg_arc = 0.0

        num = 0
        t_emb = []
        t_list = []
        for i, data in enumerate(pbar, 0):
            inputs, label = data
            optimizer_adam.zero_grad()
            inputs = inputs.to(device)
            label = label.to(device)
            cnt += 1
            arc_output, emb = model(inputs, label)
            t_emb.extend(emb.detach().cpu().numpy())
            t_list.append(label.detach().cpu().numpy().tolist())
            num += inputs.size(0)
            # res = accuracy(class_output, label)
            # acc_sum += res
            loss_arc = softmax_loss(arc_output, label)
            avg_arc += loss_arc

            #loss_fl = criterion_fl(class_output, label) * 0.1
            #loss_triplet = TripletLoss(margin=0.3)(emb, label) * 0.1
            loss = (loss_arc)

            loss.backward()
            inputs.cpu()
            label.cpu()
            # if epoch < 10:
            optimizer_adam.step()
            running_loss += loss.item()
            pbar.set_description(
                "training: Epoch:[%d/ %d] running loss %.5f, arc_loss %.3f" % (epoch,
                                                                               opt.epochs,
                                                                               running_loss / cnt,
                                                                               avg_arc / cnt,
                                                                               # acc_sum / num
                                                                               ))
        # if (epoch < 10) & (epoch != 0):
        lrschedule_adam.step()

        pbar = tqdm(valid_loader)

        valid_loss = 0.0
        avg_loss = 0.0
        sum_acc = 0.0
        model.eval()
        avg_arc = 0.0
        num = 0
        with torch.no_grad():
            cnt = 0
            for i, data in enumerate(pbar, 0):
                inputs, label = data
                cnt += 1
                inputs = inputs.to(device)
                label = label.to(device)
                arc_output, emb = model(inputs, label)
                # res = accuracy(class_output, label)
                # sum_acc += res
                num += inputs.size(0)
                #c_loss = criterion_fl(class_output, label) * 0.1
                arc_loss = softmax_loss(arc_output, label)
                #loss_triplet = TripletLoss(margin=0.3)(emb, label) * 0.1
                avg_arc += arc_loss
                valid_loss += (arc_loss)
                avg_loss = valid_loss / cnt
                inputs.cpu()
                label.cpu()
                pbar.set_description(
                    "validating loss: %.3f, valid arc loss %.3f" % (avg_loss, avg_arc / cnt))
        acc = one_epoch_acc()
        if best_valid < acc:
            best_valid = acc
        else:
            still += 1
            if still == PATIENCE:
                # KNN_fit_test(model, train_loader, valid_loader, device)
                break

        checkpoint = {
            "net": model.state_dict(),
            'optimizer_adam': optimizer_adam.state_dict(),
            "epoch": epoch,
            "lrschedule_adam": lrschedule_adam
        }
        # if not os.path.isdir("./version"):
        # os.mkdir("./version" )
        torch.save(checkpoint, './model_%s_%f.pth' % (str(epoch), acc))

        print(f" epoch index: {epoch}")
        print(
            "------------------------------------------------------------------------------------------------------")


if __name__ == '__main__':
    trainwithvalid()
