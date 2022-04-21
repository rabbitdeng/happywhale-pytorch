import numpy as np
import torch.utils.data as data
import os
import PIL.Image as Image

from config import parser

from torchvision.transforms import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

opt = parser.parse_args()

transform = {

    "train": A.Compose([

        # pil.image to tensor
        # A.HorizontalFlip(p=1.0),
        # A.VerticalFlip(p=0.5),
        # A.CenterCrop(32,32),
        #A.Resize(384,384),

        A.OneOf([A.MotionBlur(),
                 A.MedianBlur(),
                 A.Blur()], p=0.1),

        A.OneOf([A.GaussNoise(),
                 A.ToGray()], p=0.1),

        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=30, p=1),
        A.RandomBrightnessContrast(p=0.1, brightness_limit=0.1, contrast_limit=0.1),
        #A.Normalize(mean=[0.485, 0.456, 0.406],
        #            std=[0.229, 0.224, 0.225], ),
        #ToTensorV2(),  # torchvision.transforms.Normalize(mean, std, inplace=False)
    ]),
    # tensor([0.4560, 0.4245, 0.3905], device='cuda:0') tensor([0.2755, 0.2671, 0.2602], device='cuda:0')
    "valid": A.Compose([
        A.Resize(384,384),
        # A.CenterCrop(32,32),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225], ),
        ToTensorV2(),  # torchvision.transforms.Normalize(mean, std, inplace=False)
    ])
}


class WhaleDataset(data.Dataset):

    def __init__(self, df, mode, transform=None):

        if mode == 'train':

            # print(train_csv.head())
            train_img = df.image
            # print(train_img)

            labels = df.individual_key
            imgs = []
            for file, label in zip(train_img, labels):
                imgs.append([file, label])

            self.imgs = imgs

            # self.transform = transform
            self.mode = mode
        if mode == 'train_noaug':

            # print(train_csv.head())
            train_img = df.image
            # print(train_img)

            labels = df.individual_key
            imgs = []
            for file, label in zip(train_img, labels):
                imgs.append([file, label])

            self.imgs = imgs

            # self.transform = transform
            self.mode = mode
        if mode == 'valid':
            # print(train_csv.head())
            valid_img = df.image
            # print(train_img)

            labels = df.individual_key
            imgs = []
            for file, label in zip(valid_img, labels):
                imgs.append([file, label])

            self.imgs = imgs
            self.mode = mode
        if mode == 'test':
            train_img = df.image
            imgs = []
            for file in train_img:
                imgs.append(file)

            self.imgs = imgs

            # self.transform = transform
            self.mode = mode

    def __getitem__(self, idx: int):

        # x是jpg格式，label是png
        if self.mode == "train":
            x_path, label = self.imgs[idx]
            pilimage = Image.open(os.path.join(opt.train_path, x_path)).convert("RGB")
            pilimage = np.array(pilimage)
            pilimage = transform["train"](image=pilimage)

            return pilimage["image"], int(label)

        if self.mode == "train_noaug":
            x_path, label = self.imgs[idx]
            pilimage = Image.open(os.path.join(opt.train_path, x_path)).convert("RGB")
            pilimage = np.array(pilimage)
            pilimage = transform["valid"](image=pilimage)

            return pilimage["image"], int(label)
        if self.mode == "valid":
            x_path, label = self.imgs[idx]
            pilimage = Image.open(os.path.join(opt.valid_path, x_path)).convert("RGB")
            pilimage = np.array(pilimage)
            pilimage = transform["valid"](image=pilimage)

            return pilimage["image"], int(label)

        if self.mode == "test":
            x_path = self.imgs[idx]
            pilimage = Image.open(os.path.join(opt.test_path, x_path)).convert("RGB")
            pilimage = np.array(pilimage)
            pilimage = transform["valid"](image=pilimage)

            return pilimage["image"]

        # if self.transform is not None:
        # img_x = self.transform(img_x)
        # plt.imshow(pilimage)
        # plt.show()

    def __len__(self):
        return len(self.imgs)
