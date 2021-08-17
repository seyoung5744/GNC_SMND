# %%
import os            
import pandas as pd
# %%
class_names = ['Corona', 'Noise', 'Surface', 'Void']
# %%
dataset = "split_data/prpd/train/"
names = []
labels = []

for folder in os.listdir(dataset):
    label = folder
    
    for file in os.listdir(os.path.join(dataset, folder)):
        file = file[:-15]
        names.append(file)
        labels.append(label)
# %%
df = pd.DataFrame({"name":names, "label":labels}, index=None)
df
# %%
df.to_csv("split_data/train.csv", index=None, encoding="utf-8-sig")
# %%
dataset = "split_data/prpd/test/"
names = []
labels = []

for folder in os.listdir(dataset):
    label = folder
    
    for file in os.listdir(os.path.join(dataset, folder)):
        file = file[:-15]
        names.append(file)
        labels.append(label)
# %%
df = pd.DataFrame({"name":names, "label":labels}, index=None)
df
# %%
df.to_csv("split_data/test.csv", index=None, encoding="utf-8-sig")