#import packages
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory, image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img,ImageDataGenerator

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Dense,Conv2D,Dropout,GlobalAveragePooling2D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,Callback

print(tf.__version__)

from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16, VGG19 
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB2 


def set_gpu_environment():
    # tf.config.set_soft_device_placement(True)  # 다른 장치로 할당할 수 있도록 함.
    gpus = tf.config.experimental.list_physical_devices("GPU") # 사용 가능한 gpu 여부 
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU') # 텐서플로가 첫 번째 GPu만 사용하도록 제한
            tf.config.experimental.set_memory_growth(gpus[1], True) # GPU 메모리 제한하기
            
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
            print(e)
        
        
SEED = 1234
def set_seed(SEED):
    random.seed(SEED) # Python
    np.random.seed(SEED) # numpy

    tf.random.set_seed(SEED) # tensorflow
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print("set seed")