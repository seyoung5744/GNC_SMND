# %%
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle                                                        
import tensorflow as tf
import cv2
import datetime    
# %%
base_path = os.getcwd()
print(base_path)
# %%
class_names = ['Corona', 'Noise', 'Surface', 'Void']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (256, 256)
# %%
def load_data():
    datasets = ['raw/images/', 'data_images/test/']
    output = []
    
    for dataset in datasets:
        imgname = []
        images = []
        labels = []
        print("Loading {}".format(dataset))

        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            for file in tqdm(os.listdir(os.path.join(dataset, folder))):    
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE) 
                
                imgname.append(file)
                images.append(image)
                labels.append(label)

        imgname = np.array(imgname)        
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((imgname, images, labels))

    return output
# %%
(train_imgname, train_images, train_labels), (test_imgname, test_images, test_labels) = load_data()
# %%
train_imgname, train_images, train_labels = shuffle(train_imgname, train_images, train_labels, random_state=25)
# %%
train_images = train_images / 255.0 
test_images = test_images / 255.0
# %%
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (256, 256, 3)), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(4, activation=tf.nn.softmax)
])
# %%
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
# %%
history = model.fit(train_images, train_labels, batch_size=10, epochs=10, validation_split=0.2)
# %%
model.save("cnn_image_all.h5")
# %%
result = model.evaluate(test_images, test_labels)
print(result)
# %%
predictions = model.predict(test_images)
pred_labels = np.argmax(predictions, axis = 1)
# %%
df = pd.DataFrame(test_imgname, columns=["imgname"])
# %%
df["Corona"] = predictions[: , 0]
df["Noise"] = predictions[: , 1]
df["Surface"] = predictions[: , 2]
df["Void"] = predictions[: , 3]
df["predict"] = pred_labels
# %%
filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
df.to_csv(filename+".csv", index=False, encoding="utf-8-sig")
# %%
