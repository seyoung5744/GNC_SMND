#import packages
import os
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import VGG16, VGG19 
from tensorflow.keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import EfficientNetB2 


def set_gpu_environment(gpu_num=1):
    # tf.config.set_soft_device_placement(True)  # 다른 장치로 할당할 수 있도록 함.
    gpus = tf.config.experimental.list_physical_devices("GPU") # 사용 가능한 gpu 여부 
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU') # 텐서플로가 첫 번째 GPu만 사용하도록 제한
            tf.config.experimental.set_memory_growth(gpus[gpu_num], True) # GPU 메모리 제한하기
            
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
    
def model_net(model, image_shape=(260, 260, 3)) :
    if model == "Xception":
        model = Xception(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.xception.preprocess_input

    elif model == "VGG16":
        model = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.vgg16.preprocess_input

    elif model == "VGG19":
        model = VGG19(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.vgg19.preprocess_input
              
    elif model == "ResNet50":
        model = ResNet50(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input =  tf.keras.applications.resnet.preprocess_input
              
    elif model == "ResNet50V2":
        model = ResNet50V2(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.resnet.preprocess_input
              
    elif model == "ResNet101":
        model = ResNet101(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.resnet.preprocess_input
              
    elif model == "ResNet101V2":
        model = ResNet101V2(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.resnet.preprocess_input
              
    elif model == "ResNet152":
        model = ResNet152(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.resnet.preprocess_input
              
    elif model == "ResNet152V2":
        model = ResNet152V2(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.resnet.preprocess_input
              
    elif model == "InceptionV3":
        model = InceptionV3(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.inception_v3.preprocess_input
              
    elif model == "InceptionResNetV2":
        model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
              
    elif model == "MobileNet":
        model = MobileNet(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input
              
    elif model == "MobileNetV2":
        model = MobileNetV2(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
              
    elif model == "DenseNet121":
        model = DenseNet121(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.densenet.preprocess_input
              
    elif model == "DenseNet169":
        model = DenseNet169(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.densenet.preprocess_input
              
    elif model == "DenseNet201":
        model = DenseNet201(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.densenet.preprocess_input
              
    elif model == "EfficientNetB2":
        model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=image_shape)
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input

    return model, preprocess_input