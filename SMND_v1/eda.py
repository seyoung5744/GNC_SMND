# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.4)
import cv2
# %%
df_train = pd.read_csv("split_data/train.csv")
sns.countplot(df_train["label"])
df_train["label"].value_counts()
# %%
print("trainset : {}".format(len(df_train)))
# %%
df_test = pd.read_csv("split_data/test.csv")
sns.countplot(df_test["label"])
df_test["label"].value_counts()
# %%
print("testset : {}".format(len(df_test)))
# %%
prpd_imgs = []
prpd_path = "split_data/prpd/train/"
prpd_type = "[128_128변환].csv"

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('black')
fig.suptitle("csv to image", fontsize=16, color="white")
for i in range(10):
    index = i
    prpd_img_path = prpd_path + df_train["label"][index] + "/" + df_train["name"][index] + prpd_type
    prpd_img =  np.loadtxt(prpd_img_path, delimiter=",", dtype=np.float32)
    prpd_img = np.reshape(prpd_img, (256, 256, 1))
    plt.subplot(2, 5, i+1)
    plt.axis("off")
    plt.imshow(prpd_img, cmap=plt.cm.binary)
    plt.title(df_train["label"][index], color="white")
plt.show()
# %%
psa_imgs = []
psa_path = "split_data/psa/train/"
psa_type = "[PSA변환].csv"

fig = plt.figure(figsize=(10,5))
fig.patch.set_facecolor('black')
fig.suptitle("csv to image", fontsize=16, color="white")
for i in range(10):
    index = i + 60
    psa_img_path = psa_path + df_train["label"][index] + "/" + df_train["name"][index] + psa_type
    psa_img =  np.loadtxt(psa_img_path, delimiter=",", dtype=np.float32)
    psa_img = np.reshape(psa_img, (256, 256, 1))
    plt.subplot(2, 5, i+1)
    plt.axis("off")
    plt.imshow(psa_img)
    plt.title(df_train["label"][index], color="white")
plt.show()
# %%
image_prpd = cv2.imread("split_data/prpd_images/train/Corona/SMND_345kV_EBG_A_S_0A_62_20190515132500.dat.png")
image_prpd = cv2.cvtColor(image_prpd, cv2.COLOR_RGB2BGR)
plt.axis("off")
plt.imshow(image_prpd)
plt.show()

# %%
