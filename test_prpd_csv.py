import os            
import pandas as pd
import numpy as np
import datetime
from keras.models import load_model

class_names = ['Corona', 'Noise', 'Surface', 'Void']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

dataset = "split_data/prpd/test/"
test_imgnames = []
test_images = []
test_labels = []

for folder in os.listdir(dataset):
    label = class_names_label[folder]
    
    for file in os.listdir(os.path.join(dataset, folder)):
        img_path = os.path.join(os.path.join(dataset, folder), file)

        image = np.loadtxt(img_path, delimiter=",", dtype=np.float32)
        image = np.reshape(image, (256, 256, 1))

        test_imgnames.append(file)
        test_images.append(image)
        test_labels.append(label)

(test_imgnames, test_images, test_labels) = (np.array(test_imgnames), np.array(test_images), np.array(test_labels, dtype="int32"))

model = load_model('prpd_csv.h5')

predictions = model.predict(test_images).round(3)
pred_labels = np.argmax(predictions, axis = 1)

df = pd.DataFrame(test_imgnames, columns=["imgname"])
df["label"] = [class_names[label] for label in test_labels]
df[class_names] = predictions
df["predict"] = [class_names[label] for label in pred_labels]

acc = len(df.loc[df['label'] == df["predict"]])/len(df)
print("Accuracy : {}".format(acc))

filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
df.to_csv(filename+".csv", index=False, encoding="utf-8-sig")