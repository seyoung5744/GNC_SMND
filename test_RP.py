import os            
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import load_model
import cv2

class_names = ['Noise', 'PD']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

# dataset = "split_data/prpd/test/"
datasets = "./images/split_data/RP/test/"
test_imgnames = []
test_images = []
test_labels = []

IMAGE_SIZE = (256, 256)
print("class_names_label",class_names_label)
for folder in os.listdir(datasets):
    label = class_names_label[folder]
    print(label)
    for file in os.listdir(os.path.join(datasets, folder)):
        img_path = os.path.join(os.path.join(datasets, folder), file)
        print(img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, IMAGE_SIZE)     
        image = np.reshape(image, (256, 256, 3))

        test_imgnames.append(file)
        test_images.append(image)
        test_labels.append(label)

(test_imgnames, test_images, test_labels) = (np.array(test_imgnames), np.array(test_images), np.array(test_labels, dtype="int32"))
print("test_imgnames", test_imgnames)
model = load_model('./model/RP.h5')

predictions = model.predict(test_images).round(3)
pred_labels = np.argmax(predictions, axis = 1)

print(predictions)
print(pred_labels)

df = pd.DataFrame(test_imgnames, columns=["imgname"])
df["label"] = [class_names[label] for label in test_labels]
df[class_names] = predictions
df["predict"] = [class_names[label] for label in pred_labels]

acc = len(df.loc[df['label'] == df["predict"]])/len(df)
print("Accuracy : {}".format(acc))

filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
df.to_csv("results/RP_" + filename+".csv", index=False, encoding="utf-8-sig")