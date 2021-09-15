import os            
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import load_model

class_names = ['Corona', 'Noise', 'Surface', 'Void']

datasets = "./data/test data/"
test_imgnames = []
test_images = []


for file in os.listdir(datasets):
    img_path = os.path.join(datasets, file) # ./data/split_data/05. 표준데이터(PRPD)/test/Corona\SMND_345kV_EBG_A_S_0A_62_20190515133600.dat[PRPD변환].csv

    image = np.loadtxt(img_path, delimiter=",", dtype=np.float32)
    image = np.reshape(image, (256, 256, 1))

    test_imgnames.append(file)
    test_images.append(image)


(test_imgnames, test_images) = (np.array(test_imgnames), np.array(test_images))


model = load_model('prpd_csv.h5')

predictions = model.predict(test_images).round(3)
pred_labels = np.argmax(predictions, axis = 1)

# print(test_imgnames)
print(predictions)
# print(pred_labels) [3 3 3 3 3 3]


labeling = [class_names[label] for label in pred_labels]

result = {}
for i in range(len(test_imgnames)):
    result[test_imgnames[i]] = [labeling[i], predictions[i]]
    
print(result)