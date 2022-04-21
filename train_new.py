# if __name__ == '_main_':

import os
import glob
import pandas as pd
import numpy as np
import logging
import timm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Compose, Lambda, Normalize, AutoAugment, AutoAugmentPolicy

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as LP
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s" % pytorch_metric_learning.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_NAME = 'tf_efficientnet_b4_ns'
N_CLASSES = 15587
OUTPUT_SIZE = 1792
EMBEDDING_SIZE = 512
N_EPOCH = 15
BATCH_SIZE = 16
ACCUMULATION_STEPS = int(256 / BATCH_SIZE)
MODEL_LR = 1e-3
PCT_START = 0.3
PATIENCE = 5
N_WORKER = 2
N_NEIGHBOURS = 750

TRAIN_DIR = './tmp/train_images'
TEST_DIR = './tmp/test_images'
LOG_DIR = "./logs/{}".format(MODEL_NAME)
MODEL_DIR = "./models/{}".format(MODEL_NAME)


class HappyWhaleDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            image_dir: str,
            return_labels=True,
    ):
        self.df = df
        self.images = self.df["image"]
        self.image_dir = image_dir
        self.image_transform = Compose(
            [
                AutoAugment(AutoAugmentPolicy.IMAGENET),
                Lambda(lambda x: x / 255),

            ]
        )
        self.return_labels = return_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = os.path.join(self.image_dir, self.images.iloc[idx])
        image = read_image(path=image_path)
        image = self.image_transform(image)

        if self.return_labels:
            label = self.df['label'].iloc[idx]
            return image, label
        else:
            return image


df = pd.read_csv('train_new.csv')
df.head()

df['label'] = df.groupby('individual_id').ngroup()
df['label'].describe()

valid_proportion = 0.1

valid_df = df.sample(frac=valid_proportion, replace=False, random_state=1).copy()
train_df = df[~df['image'].isin(valid_df['image'])].copy()

print(train_df.shape)
print(valid_df.shape)

train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
train_dataset = HappyWhaleDataset(df=train_df, image_dir=TRAIN_DIR, return_labels=True)
len(train_dataset)
valid_dataset = HappyWhaleDataset(df=valid_df, image_dir=TRAIN_DIR, return_labels=True)
len(valid_dataset)

dataset_dict = {"train": train_dataset, "val": valid_dataset}

trunk = timm.create_model(MODEL_NAME, pretrained=True)
trunk.classifier = common_functions.Identity()
trunk = trunk.to(device)
trunk_optimizer = optim.SGD(trunk.parameters(), lr=MODEL_LR, momentum=0.9)
trunk_schedule = optim.lr_scheduler.OneCycleLR(
    trunk_optimizer,
    max_lr=MODEL_LR,
    total_steps=N_EPOCH * int(len(train_dataset) / BATCH_SIZE),
    pct_start=PCT_START
)
embedder = nn.Linear(OUTPUT_SIZE, EMBEDDING_SIZE).to(device)
embedder_optimizer = optim.SGD(trunk.parameters(), lr=MODEL_LR, momentum=0.9)
embedder_schedule = optim.lr_scheduler.OneCycleLR(
    embedder_optimizer,
    max_lr=MODEL_LR,
    total_steps=N_EPOCH * int(len(train_dataset) / BATCH_SIZE),
    pct_start=PCT_START
)

loss_func = losses.ArcFaceLoss(num_classes=N_CLASSES, embedding_size=EMBEDDING_SIZE).to(device)
loss_optimizer = optim.SGD(trunk.parameters(), lr=MODEL_LR, momentum=0.9)
loss_schedule = optim.lr_scheduler.OneCycleLR(
    loss_optimizer,
    max_lr=MODEL_LR,
    total_steps=N_EPOCH * int(len(train_dataset) / BATCH_SIZE),
    pct_start=PCT_START
)

record_keeper, _, _ = LP.get_record_keeper(LOG_DIR)
hooks = LP.get_hook_container(record_keeper, primary_metric='mean_average_precision')

tester = testers.GlobalEmbeddingSpaceTester(
    end_of_testing_hook=hooks.end_of_testing_hook,
    accuracy_calculator=AccuracyCalculator(
        include=['mean_average_precision'],
        device=torch.device("cpu"),
        k=5),
    dataloader_num_workers=N_WORKER,
    batch_size=BATCH_SIZE
)
end_of_epoch_hook = hooks.end_of_epoch_hook(
    tester,
    dataset_dict,
    MODEL_DIR,
    test_interval=1,
    patience=PATIENCE,
    splits_to_eval=[('val', ['train'])]
)


class HappyTrainer(trainers.MetricLossOnly):
    def __init__(self, *args, accumulation_steps=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulation_steps = accumulation_steps

    def forward_and_backward(self):
        self.zero_losses()
        self.update_loss_weights()
        self.calculate_loss(self.get_batch())
        self.loss_tracker.update(self.loss_weights)
        self.backward()
        self.clip_gradients()
        if ((self.iteration + 1) % self.accumulation_steps == 0) or (
                (self.iteration + 1) == np.ceil(len(self.dataset) / self.batch_size)):
            self.step_optimizers()
            self.zero_grad()

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        with torch.cuda.amp.autocast():
            embeddings = self.compute_embeddings(data)
            indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
            self.losses["metric_loss"] = self.maybe_get_metric_loss(
                embeddings, labels, indices_tuple
            )


trainer = HappyTrainer(
    models={"trunk": trunk, "embedder": embedder},
    optimizers={"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer,
                "metric_loss_optimizer": loss_optimizer},
    batch_size=BATCH_SIZE,
    loss_funcs={"metric_loss": loss_func},
    mining_funcs={},
    dataset=train_dataset,
    dataloader_num_workers=N_WORKER,
    end_of_epoch_hook=end_of_epoch_hook,
    lr_schedulers={
        'trunk_scheduler_by_iteration': trunk_schedule,
        'embedder_scheduler_by_iteration': embedder_schedule,
        'metric_loss_scheduler_by_iteration': loss_schedule,
    },
    accumulation_steps=ACCUMULATION_STEPS
)

# TRAINing
trainer.train(num_epochs=N_EPOCH)

logging.getLogger().setLevel(logging.WARNING)

best_trunk_weights = glob.glob('../models/{}/trunk_best*.pth'.format(MODEL_NAME))[0]
trunk.load_state_dict(torch.load(best_trunk_weights))

best_embedder_weights = glob.glob('../models/{}/embedder_best*.pth'.format(MODEL_NAME))[0]
embedder.load_state_dict(torch.load(best_embedder_weights))

inference_model = InferenceModel(
    trunk=trunk,
    embedder=embedder,
    normalize_embeddings=True,
)
inference_model.train_knn(train_dataset)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKER,
                              pin_memory=True)

valid_labels_list = []
valid_distance_list = []
valid_indices_list = []

for images, labels in tqdm(valid_dataloader):
    distances, indices = inference_model.get_nearest_neighbors(images, k=N_NEIGHBOURS)
    valid_labels_list.append(labels)
    valid_distance_list.append(distances)
    valid_indices_list.append(indices)

valid_labels = torch.cat(valid_labels_list, dim=0).cpu().numpy()
valid_distances = torch.cat(valid_distance_list, dim=0).cpu().numpy()
valid_indices = torch.cat(valid_indices_list, dim=0).cpu().numpy()
new_whale_idx = -1

train_labels = train_df['individual_id'].unique()
train_idx_lookup = train_df['individual_id'].copy().to_dict()
train_idx_lookup[-1] = 'new_individual'

valid_class_lookup = valid_df.set_index('label')['individual_id'].copy().to_dict()

thresholds = [np.quantile(valid_distances, q=q) for q in np.arange(0, 1.0, 0.01)]

results = []

for threshold in tqdm(thresholds):

    prediction_list = []
    running_map = 0

    for i in range(len(valid_distances)):

        pred_knn_idx = valid_indices[i, :].copy()
        insert_idx = np.where(valid_distances[i, :] > threshold)

        if insert_idx[0].size != 0:
            pred_knn_idx = np.insert(pred_knn_idx, np.min(insert_idx[0]), new_whale_idx)

        predicted_label_list = []

        for predicted_idx in pred_knn_idx:
            predicted_label = train_idx_lookup[predicted_idx]
            if len(predicted_label_list) == 5:
                break
            if (predicted_label == 'new_individual') | (predicted_label not in predicted_label_list):
                predicted_label_list.append(predicted_label)

        gt = valid_class_lookup[valid_labels[i]]

        if gt not in train_labels:
            gt = "new_individual"

        precision_vals = []

        for j in range(5):
            if predicted_label_list[j] == gt:
                precision_vals.append(1 / (j + 1))
            else:
                precision_vals.append(0)

        running_map += np.max(precision_vals)

    results.append([threshold, running_map / len(valid_distances)])

results_df = pd.DataFrame(results, columns=['threshold', 'map5'])

results_df = results_df.sort_values(by='map5', ascending=False).reset_index(drop=True)
results_df.head(5)

threshold = results_df.loc[0, 'threshold']
# inference test

combined_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
combined_dataset = HappyWhaleDataset(df=combined_df, image_dir=TRAIN_DIR, return_labels=True)
len(combined_dataset)

inference_model.train_knn(combined_dataset)

test_df = pd.read_csv('../input/happy-whale-and-dolphin/sample_submission.csv')

test_dataset = HappyWhaleDataset(df=test_df, image_dir=TEST_DIR, return_labels=False)
len(test_dataset)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKER, pin_memory=True)

test_distance_list = []
test_indices_list = []

for images in tqdm(test_dataloader):
    distances, indices = inference_model.get_nearest_neighbors(images, k=N_NEIGHBOURS)
    test_distance_list.append(distances)
    test_indices_list.append(indices)

test_distances = torch.cat(test_distance_list, dim=0).cpu().numpy()
test_indices = torch.cat(test_indices_list, dim=0).cpu().numpy()

combined_idx_lookup = combined_df['individual_id'].copy().to_dict()
combined_idx_lookup[-1] = 'new_individual'

results = []

prediction_list = []

for i in range(len(test_distances)):

    pred_knn_idx = test_indices[i, :].copy()
    insert_idx = np.where(test_distances[i, :] > threshold)

    if insert_idx[0].size != 0:
        pred_knn_idx = np.insert(pred_knn_idx, np.min(insert_idx[0]), new_whale_idx)

    predicted_label_list = []

    for predicted_idx in pred_knn_idx:
        predicted_label = combined_idx_lookup[predicted_idx]
        if len(predicted_label_list) == 5:
            break
        if (predicted_label == 'new_individual') | (predicted_label not in predicted_label_list):
            predicted_label_list.append(predicted_label)

    prediction_list.append(predicted_label_list)

prediction_df = pd.DataFrame(prediction_list)
prediction_df.head()

prediction_df['predictions'] = prediction_df[0].astype(str) + ' ' + prediction_df[1].astype(str) + ' ' + prediction_df[
    2].astype(str) + ' ' + prediction_df[3].astype(str) + ' ' + prediction_df[4].astype(str)
prediction_df.head()

submission = pd.read_csv('./happy-whale-and-dolphin/sample_submission.csv')
submission['predictions'] = prediction_df['predictions']
submission.head(1)

submission.to_csv('submission.csv', index=False)
