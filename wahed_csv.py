import pandas as pd
import os
# use for generate new washed csv file of current train images left.
pics_dir = "cropped_train_images/cropped_train_images"
df = pd.read_csv("train_encoded.csv")
df_new = df
list_existpic = os.listdir(pics_dir)
for idx, img in enumerate(df.image):
    if img not in list_existpic:
        print("drop ", img)
        df_new = df_new.drop(index=idx)

df_new.to_csv("washed_train_encoded.csv", index=False)
