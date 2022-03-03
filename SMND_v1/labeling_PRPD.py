# %%
import os            
import pandas as pd
# %%
class_names = ['Corona', 'Noise', 'Surface', 'Void']
# %%
dataset = "./data/split_data/05. 표준데이터(PRPD)/train/"
names = []
labels = []

for folder in os.listdir(dataset):
    label = folder
    label = label.split(" ")[1]

    for file in os.listdir(os.path.join(dataset, folder)):
        file = file[:-15]
        names.append(file)
        labels.append(label)
# %%
df = pd.DataFrame({"name":names, "label":labels}, index=None)
df
# %%
df.to_csv("./data/split_data/train.csv", index=None, encoding="utf-8-sig")
# %%
dataset = "./data/split_data/05. 표준데이터(PRPD)/test/"
names = []
labels = []

for folder in os.listdir(dataset):
    label = folder
    label = label.split(" ")[1]

    for file in os.listdir(os.path.join(dataset, folder)):
        file = file[:-15]
        names.append(file)
        labels.append(label)
# %%
df = pd.DataFrame({"name":names, "label":labels}, index=None)
df
# %%
df.to_csv("./data/split_data/test.csv", index=None, encoding="utf-8-sig")