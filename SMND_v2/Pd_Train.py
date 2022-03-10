#import packages

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import load_model

"""
model_name : 불러올 모델명
trainable 
    False : 전이학습 O
    True  : 전이학습 X
"""
class PdTrain:
    def __init__(self, model_name, trainable=False):
        self.model_net, self.preprocess_input = model_net(model_name)

        self.model_net.trainable = trainable
   
        inputs = self.model_net.input
        x = self.preprocess_input(inputs)
        x = self.model_net.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(4, activation='softmax')(x) # Dense

        self.model = Model(inputs=inputs, outputs=outputs)  
        
        
    def train(self, train_data, valid_data=None, learning_rate=0.0001, epoch=100):
        print(self.model)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy', 
                     metrics=['accuracy']) 

        best_acc_pth = "./test model/" + self.model_net.name + "/" + self.model_net.name + "_best_acc.h5"
        best_loss_pth = "./test model/" + self.model_net.name + "/" + self.model_net.name + "_best_loss.h5"

        cp_best_acc = ModelCheckpoint(best_acc_pth, monitor= 'val_accuracy', verbose = 0, save_best_only = True, mode = 'auto')
        cp_best_loss = ModelCheckpoint(best_loss_pth, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto')

        history = self.model.fit(train_data,
                            validation_data=valid_data, 
                            epochs=epoch,
                            callbacks = [cp_best_acc, cp_best_loss],
                            verbose = 2)
        
        train_acc = history.history['accuracy']
        train_loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']  
        
        '''
        dictionary 형태로 반환, value는 list 형태
        '''
        return {'accuracy': train_acc, 'val_accuracy': val_acc, 'loss': train_loss, 'val_loss': val_loss}
        
    def evaluate(self, best_acc_pth, best_loss_pth, train_data):
        # best_acc
        self.model.load_weights(best_acc_pth)
        loss1, accuracy1 = model.evaluate(train_data)

        # best_loss
        self.model.load_weights(best_loss_pth)
        loss2, accuracy2 = model.evaluate(train_data)
        
        '''
        화면에 나타낼 떄
        1차적으로 accuracy가 큰거 먼저 
        if, accuracy가 동일하다면 loss가 작은것을 화면에 출력
        '''
        return {"best accuracy model" :(loss1, accuracy1), "best loss model" :(loss2, accuracy2)}
    
    def predict(self, model_path, test_data):
        model = load_model(model_path)
        yhat_test = model.predict(test_data).round(4)
        
        return yhat_test


    def get_classification_report(self, true_label, predict_label):
        return classification_report(true_label, predict_label, target_names=["Corona","Noise","Surface","Void"])
        
    def get_confusion_matrix(self, true_label, predict_label):
        '''
        반환시 다음과 같은 형태 띔.
        [[17  0  0  0]
         [ 0  9  0  2]
         [ 0  0 27  2]
         [ 0  0  3 38]]
        '''
         return confusion_matrix(true_label, predict_label)
        