# %%
import os        
import pandas as pd
import numpy as np
import datetime
from keras.models import load_model
# %%
class_names = ['Corona', 'Noise', 'Surface', 'Void']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (256, 256)

df = pd.read_csv("split_data/test.csv")
# %%
labels = []

prpd_imgs = []
prpd_path = "split_data/prpd/test/"
prpd_type = "[128_128변환].csv"

psa_imgs = []
psa_path = "split_data/psa/test/"
psa_type = "[PSA변환].csv"

for name, label in zip(df["name"], df["label"]):
    prpd_img_path = prpd_path + label + "/" + name + prpd_type
    prpd_img =  np.loadtxt(prpd_img_path, delimiter=",", dtype=np.float32)
    prpd_img = np.reshape(prpd_img, (256, 256, 1))

    psa_img_path = psa_path + label + "/" + name + psa_type
    psa_img =  np.loadtxt(psa_img_path, delimiter=",", dtype=np.float32)
    psa_img = np.reshape(psa_img, (256, 256, 1))
    
    prpd_imgs.append(prpd_img)
    psa_imgs.append(psa_img)
    labels.append(class_names_label[label])

(prpd_imgnames, prpd_imgs, prpd_labels) = (np.array(df["name"]), np.array(prpd_imgs), np.array(labels, dtype="int32"))
(psa_imgnames, psa_imgs, psa_labels) = (np.array(df["name"]), np.array(psa_imgs), np.array(labels, dtype="int32"))


# %%
prpd_model = load_model("prpd_csv.h5")
psa_model = load_model("psa_csv.h5")
# %%
prpd_predict = prpd_model(prpd_imgs).numpy().round(3)
psa_predict = psa_model(psa_imgs).numpy().round(3)

predict = (prpd_predict * 0.5 + psa_predict * 0.5).round(3)
pred_labels = np.argmax(predict, axis=1)
# %%
class_names_prpd = ['prpd_Corona', 'prpd_Noise', 'prpd_Surface', 'prpd_Void']
class_names_psa = ['psa_Corona', 'psa_Noise', 'psa_Surface', 'psa_Void']

result = df.copy()
result[class_names_prpd] = prpd_predict
result[class_names_psa] = psa_predict
result[class_names] = predict

predict_labels = [class_names[label] for label in pred_labels]
result["predict"] = predict_labels

unknown = np.array(np.where(np.max(predict, axis=1) < 0.6)[0]).tolist()
#error = np.array(np.where(np.max(predict.numpy().round(2), axis=1) == 1)[0]).tolist()
#result["predict"][unknown] = "Unknown"
#result["predict"][error] = "Error"
# %%
acc = len(result.loc[result['label'] == result["predict"]])/len(result)
print("Accuracy : {}".format(acc))
# %%
filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
result.to_csv(filename+".csv", index=False, encoding="utf-8-sig")
# %%
