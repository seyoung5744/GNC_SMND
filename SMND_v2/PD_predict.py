#import packages
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

import itertools
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.preprocessing import image_dataset_from_directory


def get_image_data(data_path, color_mode, image_size, seed = None, subset = None, shuffle=True, validation_split = None):
    if subset:
        validation_split = 0.2
    return image_dataset_from_directory(
        data_path,
        color_mode=color_mode,
        image_size=image_size,
        label_mode='categorical',
        seed=seed,
        shuffle=shuffle,
        validation_split=validation_split, 
        subset=subset
    )
    
color_mode = "rgb"
image_size = (260, 260)

test_data = get_image_data(test_dir,
                            color_mode,
                            image_size,
                            shuffle=False)

def predict_pd(model_path):

    model = load_model(model_path)
    #make prediction
    yhat_test = np.argmax(model.predict(test_data), axis=1)