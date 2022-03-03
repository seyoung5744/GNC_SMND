# %%
import os            
import pandas as pd
# %%
class_names = ['Noise', 'PD']
# %%
dataset = "./images/split_data/GASF/train/"
names = []
labels = []

for folder in os.listdir(dataset):
    label = folder
    print(label)
    # label = label.split(" ")[1]

    for file in os.listdir(os.path.join(dataset, folder)):
        print(file)
        # file = file[:-15]
        names.append(file)
        labels.append(label)
# %%
df = pd.DataFrame({"name":names, "label":labels}, index=None)
df
# %%
df.to_csv("./images/split_data/GASF/train.csv", index=None, encoding="utf-8-sig")
# %%
dataset = "./images/split_data/GASF/train/"

names = []
labels = []

for folder in os.listdir(dataset):
    label = folder
    # label = label.split(" ")[1]

    for file in os.listdir(os.path.join(dataset, folder)):
        # file = file[:-15]
        names.append(file)
        labels.append(label)
# %%
df = pd.DataFrame({"name":names, "label":labels}, index=None)
df
# %%
df.to_csv("./images/split_data/GASF/test.csv", index=None, encoding="utf-8-sig")