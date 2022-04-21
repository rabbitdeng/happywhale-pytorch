import pandas as pd
import os
import albumentations as A
import PIL.Image as Image
import numpy as np

pics_dir = "data/convert-backfintfrecords/happy-whale-and-dolphin-backfin/train_images"
output_dir = "data/convert-backfintfrecords/happy-whale-and-dolphin-backfin/aug_images_vfzero"
df_train = pd.read_csv("output/working/flip_train_folded_encoded.csv")


def gaussblur(img_path, df_new):
    pic = Image.open(os.path.join(pics_dir, img_path)).convert("RGB")
    pic = np.array(pic)
    gauss_blur = A.GlassBlur(sigma=0.2, max_delta=1, always_apply=True)
    pic_new = gauss_blur(image=pic)["image"]
    pic_new = Image.fromarray(pic_new).convert("RGB")
    pic_new.save(
        f"D:\\hw_ptlight\\data\\convert-backfintfrecords\\happy-whale-and-dolphin-backfin\\aug_images_vfzero\\G{img_path}")
    new_row = df_train.iloc[idx]
    new_row.image = f"G{img_path}"
    new_row.image_path = f" {output_dir}/G{img_path}"
    df_new = df_new.append(new_row)
    return df_new


def motionblur(img_path, df_new):
    pic = Image.open(os.path.join(pics_dir, img_path)).convert("RGB")
    pic = np.array(pic)
    m_blur = A.MotionBlur()
    pic_new = m_blur(image=pic)["image"]
    pic_new = Image.fromarray(pic_new).convert("RGB")
    pic_new.save(
        f"D:\\hw_ptlight\\data\\convert-backfintfrecords\\happy-whale-and-dolphin-backfin\\aug_images_vfzero\\mtn{img_path}")
    new_row = df_train.iloc[idx]
    new_row.image = f"mtn{img_path}"
    new_row.image_path = f" {output_dir}/mtn{img_path}"
    df_new = df_new.append(new_row)
    return df_new


def medianblur(img_path, df_new):
    pic = Image.open(os.path.join(pics_dir, img_path)).convert("RGB")
    pic = np.array(pic)
    mdnblur = A.MedianBlur(always_apply=True)
    pic_new = mdnblur(image=pic)["image"]
    pic_new = Image.fromarray(pic_new).convert("RGB")
    pic_new.save(
        f"D:\\hw_ptlight\\data\\convert-backfintfrecords\\happy-whale-and-dolphin-backfin\\aug_images_vfzero\\mdn{img_path}")
    new_row = df_train.iloc[idx]
    new_row.image = f"mdn{img_path}"
    new_row.image_path = f" {output_dir}/mdn{img_path}"
    df_new = df_new.append(new_row)
    return df_new


def rotate(img_path, df_new):
    pic = Image.open(os.path.join(pics_dir, img_path)).convert("RGB")
    pic = np.array(pic)
    rotate = A.ShiftScaleRotate(always_apply=True, rotate_limit=20)
    pic_new = rotate(image=pic)["image"]
    pic_new = Image.fromarray(pic_new).convert("RGB")
    pic_new.save(
        f"D:\\hw_ptlight\\data\\convert-backfintfrecords\\happy-whale-and-dolphin-backfin\\aug_images_vfzero\\r{img_path}")
    new_row = df_train.iloc[idx]
    new_row.image = f"r{img_path}"
    new_row.image_path = f" {output_dir}/r{img_path}"
    df_new = df_new.append(new_row)
    return df_new


def blackout(img_path, df_new):
    pic = Image.open(os.path.join(pics_dir, img_path)).convert("RGB")
    pic = np.array(pic)
    dropout = A.CoarseDropout(max_holes=8, min_holes=4, max_height=64, max_width=64, min_width=64, min_height=64, p=1)
    pic_new = dropout(image=pic)["image"]
    pic_new = Image.fromarray(pic_new).convert("RGB")
    pic_new.save(
        f"D:\\hw_ptlight\\data\\convert-backfintfrecords\\happy-whale-and-dolphin-backfin\\aug_images_vfzero\\b{img_path}")
    new_row = df_train.iloc[idx]
    new_row.image = f"b{img_path}"
    new_row.image_path = f"{output_dir}/b{img_path}"
    df_new = df_new.append(new_row)
    return df_new


def hflip(img_path, df_new):
    pic = Image.open(os.path.join(pics_dir, img_path)).convert("RGB")
    pic = np.array(pic)
    dropout = A.HorizontalFlip(always_apply=True)
    pic_new = dropout(image=pic)["image"]
    pic_new = Image.fromarray(pic_new).convert("RGB")
    pic_new.save(
        f"D:\\hw_ptlight\\data\\convert-backfintfrecords\\happy-whale-and-dolphin-backfin\\aug_images_vfzero\\h{img_path}")
    new_row = df_train.iloc[idx]
    new_row.image = f"h{img_path}"
    new_row.individual_id = "new_individual"
    new_row.image_path = f"{output_dir}/h{img_path}"
    df_new = df_new.append(new_row)
    return df_new


if __name__ == "__main__":
    df_new = pd.DataFrame()
    # list_existpic = os.listdir("augedimages")
    val_num = 3700
    val_newind = 410
    cnt = 0
    for idx, k in enumerate(df_train.kfold):
        cnt += 1
        print(cnt)
        row = df_train.iloc[idx]

        if val_newind != 0:
            if row.individual_id == 13837:
                row.kfold = 0
                val_newind -= 1
                val_num-=1
                df_new = df_new.append(row)
        if val_num != 0:
            if row.individual_id != 13837:
                val_num -= 1
                row.kfold = 0
                df_new = df_new.append(row)
        else:
            row.kfold = 1
            df_new = df_new.append(row)

        # img_path = df_train.iloc[idx].image
        # df_new = medianblur(img_path, df_new)
        # df_new = motionblur(img_path, df_new)
        # df_new = rotate(img_path, df_new)
        # df_new = gaussblur(img_path, df_new)
        # df_new = blackout(img_path, df_new)
        # df_new = hflip(img_path, df_new)
        # new_row = df_train.iloc[idx]
        # new_row.image = f"b{img_path}"
        # new_row.image_path = f"{output_dir}/b{img_path}"
        # df_new = df_new.append(new_row)
        # new_row = df_train.iloc[idx]
        # new_row.image = f"r{img_path}"
        # new_row.image_path = f"{output_dir}/r{img_path}"
        # df_new = df_new.append(new_row)
        # new_row = df_train.iloc[idx]
        # new_row.image = f"mdn{img_path}"
        # new_row.image_path = f"{output_dir}/mdn{img_path}"
        # df_new = df_new.append(new_row)
        # new_row = df_train.iloc[idx]
        # new_row.image = f"mtn{img_path}"
        # new_row.image_path = f"{output_dir}/mtn{img_path}"
        # df_new = df_new.append(new_row)
        # new_row = df_train.iloc[idx]
        # new_row.image = f"G{img_path}"
        # new_row.image_path = f"{output_dir}/G{img_path}"
        # df_new = df_new.append(new_row)

    print("finish")
    df_new.to_csv("output/working/train_newval.csv",
                  index=False)
