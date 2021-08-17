# %%
import os            
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle 
import tensorflow as tf
# %%
class_names = ['Corona', 'Noise', 'Surface', 'Void']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (256, 256)

# %%
def load_data():
    datasets = ["split_data/prpd/train/", "split_data/prpd/test/"]
    output = []

    for dataset in datasets:
        imgnames = []
        images = []
        labels = []
        print("Loading {}".format(dataset))

        for folder in os.listdir(dataset):
            label = class_names_label[folder]
            
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                img_path = os.path.join(os.path.join(dataset, folder), file)

                image = np.loadtxt(img_path, delimiter=",", dtype=np.float32)
                image = np.reshape(image, (256, 256, 1))
                
                imgnames.append(file)
                images.append(image)
                labels.append(label)
        
        imgnames = np.array(imgnames)
        images = np.array(images)
        labels = np.array(labels, dtype="int32")

        output.append((imgnames, images, labels))
    
    return output
# %%
(train_imgnames, train_images, train_labels), (test_imgnames, test_images, test_labels) = load_data()
# %%
train_imgnames, train_images, train_labels = shuffle(train_imgnames, train_images, train_labels, random_state=25)
# %%
# train_images = train_images / 255.0 
# test_images = test_images / 255.0
# %%
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (256, 256, 1)), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
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
model.save("prpd_csv.h5")
# %%
test_loss = model.evaluate(test_images, test_labels)
# %%