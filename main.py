import math
from typing import Callable, Tuple, Any
from typing import Dict
from typing import Optional
from typing import Tuple
from pathlib import Path
import warnings
import faiss
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from timm.data.transforms_factory import create_transform
from timm.optim import create_optimizer_v2

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.preprocessing import LabelEncoder
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler, StepLR

warnings.filterwarnings("ignore")
INPUT_DIR = Path("./") / "data"
OUTPUT_DIR = Path("./") / "output" / "working"

DATA_ROOT_DIR = INPUT_DIR / "convert-backfintfrecords" / "happy-whale-and-dolphin-backfin"
TRAIN_DIR = DATA_ROOT_DIR / "train_images"
TEST_DIR = DATA_ROOT_DIR / "test_images"
TRAIN_CSV_PATH = "data/working/happy-whale-and-dolphin-backfin/train.csv"
SAMPLE_SUBMISSION_CSV_PATH = DATA_ROOT_DIR / "sample_submission.csv"
PUBLIC_SUBMISSION_CSV_PATH = INPUT_DIR / "0-720-eff-b5-640-rotate" / "submission.csv"
IDS_WITHOUT_BACKFIN_PATH = INPUT_DIR / "ids-without-backfin" / "ids_without_backfin.npy"

N_SPLITS = 5

ENCODER_CLASSES_PATH = OUTPUT_DIR / "encoder_classes.npy"
TEST_CSV_PATH = OUTPUT_DIR / "test.csv"
TRAIN_CSV_ENCODED_FOLDED_PATH = OUTPUT_DIR / "train_encoded_folded.csv"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
SUBMISSION_CSV_PATH = OUTPUT_DIR / "submission.csv"

DEBUG = False

VAL_FOLD = 4.0


def get_image_path(id: str, dir: Path) -> str:
    return f"{dir / id}"


train_df = pd.read_csv(TRAIN_CSV_PATH)

train_df["image_path"] = train_df["image"].apply(get_image_path, dir=TRAIN_DIR)

encoder = LabelEncoder()
train_df["individual_id"] = encoder.fit_transform(train_df["individual_id"])
np.save(ENCODER_CLASSES_PATH, encoder.classes_)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.individual_id)):
    train_df.loc[val_, "kfold"] = fold

train_df.to_csv(TRAIN_CSV_ENCODED_FOLDED_PATH, index=False)

from albumentations import *


def strong_aug(p=0.5):
    return Compose([
        Cutout(num_holes=1, max_h_size=64, max_w_size=64, fill_value=0, p=0.2),
        #ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.2),
        #RandomBrightnessContrast(p=0.2),
        #OneOf([
        #    MotionBlur(p=0.2),
        #    MedianBlur(blur_limit=3, p=0.1),
        #    Blur(blur_limit=3, p=0.1),
        #], p=0.2),
        #OneOf([
        #    OpticalDistortion(p=0.3),
        #    GridDistortion(p=0.1),
        #    IAAPiecewiseAffine(p=0.3),
        #], p=0.2),
        # OneOf([
        #    IAASharpen(),
        #    IAAEmboss()], p=0.2),
        #OneOf([
        #   IAAAdditiveGaussianNoise(),
        #    GaussNoise(),
        #], p=0.2),
    ])


class HappyWhaleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None, is_train=False):
        self.df = df
        self.transform = transform
        self.is_train = is_train
        self.image_names = self.df["image"].values
        self.image_paths = self.df["image_path"].values
        self.targets = self.df["individual_id"].values

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # image_name = self.image_names[index]

        # image_path = self.image_paths[index]

        # image = np.array(Image.open(self.image_paths[index]).convert("RGB"))
        image = Image.open(self.image_paths[index])
        if self.transform:
            image = np.array(self.transform(image))

        if self.is_train:
            image = strong_aug(p=0.5)(image=image)["image"]

        target = self.targets[index]
        target = torch.tensor(target, dtype=torch.long)

        return {"image_name": self.image_names[index], "image": image, "target": target}

    def __len__(self) -> int:
        return len(self.df)


class LitDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_csv_encoded_folded: str,
            test_csv: str,
            val_fold: float,
            image_size: int,
            batch_size: int,
            num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_df = pd.read_csv(train_csv_encoded_folded)
        self.test_df = pd.read_csv(test_csv)
        self.batch_size = batch_size
        self.transform = create_transform(
            input_size=(self.hparams.image_size, self.hparams.image_size),
            crop_pct=1.0,

        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Split train df using fold
            train_df = self.train_df[self.train_df.kfold != self.hparams.val_fold].reset_index(drop=True)
            val_df = self.train_df[self.train_df.kfold == self.hparams.val_fold].reset_index(drop=True)

            self.train_dataset = HappyWhaleDataset(train_df, transform=self.transform, is_train=True)
            self.val_dataset = HappyWhaleDataset(val_df, transform=self.transform, is_train=False)

        if stage == "test" or stage is None:
            self.test_dataset = HappyWhaleDataset(self.test_df, transform=self.transform, is_train=False)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: HappyWhaleDataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=train,
        )


# From https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
# Added type annotations, device, and 16bit support
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            s: float,
            m: float,
            easy_margin: bool,
            ls_eps: float,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class LitModule(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            pretrained: bool,
            drop_rate: float,
            embedding_size: int,
            num_classes: int,
            arc_s: float,
            arc_m: float,
            arc_easy_margin: bool,
            arc_ls_eps: float,
            optimizer: str,
            learning_rate: float,
            weight_decay: float,
            len_train_dl: int,
            epochs: int
    ):
        super().__init__()

        self.save_hyperparameters()
        # self.fea_extra_layer = [2, 3]
        #self.fea_extra_layer = [-2, -1]
        self.model = timm.create_model(model_name, pretrained=True, drop_rate=drop_rate,
                                       #features_only=True,
                                       #out_indices=self.fea_extra_layer
                                       )
        in_features = 2304
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size))

        self.model.reset_classifier(num_classes=0, global_pool="avg")
        #self.bn = nn.Sequential(
        #    nn.Dropout(0.2),
        #    nn.AdaptiveAvgPool2d(1),
        #)
        #self.bn2 = nn.Sequential(
        #    nn.Dropout(0.2),
        #    nn.AdaptiveAvgPool2d(1),
        #)

        self.arc = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=arc_s,
            m=arc_m,
            easy_margin=arc_easy_margin,
            ls_eps=arc_ls_eps,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        embeddings = self.embedding(features.flatten(1))

        return embeddings

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.parameters(),
            opt=self.hparams.optimizer,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.learning_rate,
            steps_per_epoch=self.hparams.len_train_dl,
            epochs=self.hparams.epochs,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        images, targets = batch["image"], batch["target"]

        embeddings = self(images)
        outputs = self.arc(embeddings, targets, self.device)

        loss = self.loss_fn(outputs, targets)

        self.log(f"{step}_loss", loss)

        return loss


def train(
        train_csv_encoded_folded: str = str(TRAIN_CSV_ENCODED_FOLDED_PATH),
        test_csv: str = str(TEST_CSV_PATH),
        val_fold: float = 0.0,
        image_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 2,
        model_name: str = "tf_efficientnet_b0",
        pretrained: bool = True,
        drop_rate: float = 0.0,
        embedding_size: int = 512,
        num_classes: int = 13837,
        arc_s: float = 30.0,
        arc_m: float = 0.5,
        arc_easy_margin: bool = False,
        arc_ls_eps: float = 0.0,
        optimizer: str = "adamW",
        learning_rate: float = 3e-4,
        weight_decay: float = 5e-4,
        checkpoints_dir: str = str(CHECKPOINTS_DIR),
        accumulate_grad_batches: int = 1,
        auto_lr_find: bool = False,
        auto_scale_batch_size: bool = False,
        fast_dev_run: bool = False,
        gpus: int = 1,
        max_epochs: int = 100,
        precision: int = 16,
        stochastic_weight_avg: bool = True,
):
    pl.seed_everything(42)

    datamodule = LitDataModule(
        train_csv_encoded_folded=train_csv_encoded_folded,
        test_csv=test_csv,
        val_fold=val_fold,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    datamodule.setup()
    len_train_dl = len(datamodule.train_dataloader())

    module = LitModule(
        model_name=model_name,
        pretrained=pretrained,
        drop_rate=drop_rate,
        embedding_size=embedding_size,
        num_classes=num_classes,
        arc_s=arc_s,
        arc_m=arc_m,
        arc_easy_margin=arc_easy_margin,
        arc_ls_eps=arc_ls_eps,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        len_train_dl=len_train_dl,
        epochs=max_epochs
    )

    model_checkpoint = ModelCheckpoint(
        checkpoints_dir,
        filename=f"{model_name}_{image_size}",
        monitor="val_loss",
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=auto_scale_batch_size,
        benchmark=True,
        callbacks=[model_checkpoint, EarlyStopping(monitor="val_loss", patience=5, verbose=True)],
        deterministic=False,
        fast_dev_run=fast_dev_run,
        gpus=gpus,
        max_epochs=2 if DEBUG else max_epochs,
        precision=precision,
        stochastic_weight_avg=stochastic_weight_avg,
        limit_train_batches=0.1 if DEBUG else 1.0,
        limit_val_batches=0.1 if DEBUG else 1.0,
        amp_backend="native",
        #resume_from_checkpoint="output/working/checkpoints/convnext_base_384_in22ft1k_384.ckpt"
    )

    trainer.tune(module, datamodule=datamodule)

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    model_name = "tf_efficientnet_b6_ns"
    image_size = 384
    batch_size = 32

    train(model_name=model_name,
          image_size=image_size,
          batch_size=batch_size,
          num_workers=1,
          val_fold=VAL_FOLD

          )
