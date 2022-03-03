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
        
        return history
        
    def evaluate(self, best_acc_pth, best_loss_pth, train_data):
        # best_acc
        self.model.load_weights(best_acc_pth)
        loss1, accuracy1 = model.evaluate(train_data)

        # best_loss
        self.model.load_weights(best_loss_pth)
        loss2, accuracy2 = model.evaluate(train_data)
        
        return {"best accuracy model" :(loss1, accuracy1), "best loss model" :(loss2, accuracy2)}
    
    def predict(self, model_path, test_data):
        model = load_model(model_path)
        yhat_test = model.predict(test_data).round(4)
        
        return yhat_test


    def get_classification_report(self, true_label, predict_label):
        return classification_report(true_label, predict_label, target_names=["Corona","Noise","Surface","Void"])
        
    def get_confusion_matrix(self, true_label, predict_label):
         return confusion_matrix(true_label, predict_label)
        
    def plot_confusion_matrix(self, true_label, predict_label, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):
        cm = confusion_matrix(true_label, predict_label)
        plt.figure(figsize = (6,6))
        plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
        plt.title(title)
        plt.colorbar()
        plt.grid(False)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation = 90)
        plt.yticks(tick_marks, classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

        thresh = cm.max() / 2.
        cm = np.round(cm,2)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     fontsize = 12,
                     horizontalalignment = "center",
                     color = "white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        #     plt.savefig("./psa models/" + model_net.name + "/" + model_net.name + "_성과지표.png")
        plt.savefig("./cross correlation psa models/" + model_net.name + "/" + model_net.name + "_성과지표.png")
    