import os
import time
import datetime
import traceback

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed

import tensorflow를 위 코드뒤에 해야 오류가 안 뜬다
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


import pandas as pd
import numpy as np
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from tensorflow.keras.models import load_model


class ModelSingleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):         # Foo 클래스 객체에 _instance 속성이 없다면
            # cls._instance = super().__new__(cls)  # Foo 클래스의 객체를 생성하고 Foo._instance로 바인딩
            cls._instance = load_model('./model/prpd.h5')  # Foo 클래스의 객체를 생성하고 Foo._instance로 바인딩
            
        return cls._instance                      # Foo._instance를 리턴

    def __init__(self):
        cls = type(self)
        if not hasattr(cls, "_init"):             # Foo 클래스 객체에 _init 속성이 없다면
            cls._init = True


def prps2prpd(id, file_path, model, date_time):   
    # print("\n============= ",model, os.getpid(), " ===============================================\n")
    # print("들어온 파일 : " ,file_path, type(file_path))
    try:
        file_name = file_path.split("/")[-1]
        data = pd.read_csv(file_path, names=range(0,256))
        # print(data)
        box = np.full((256, 256), 0)

        for row, data_series in data.iteritems():
            datas = data_series.value_counts()
            for col, data in datas.iteritems():
                box[row,col] = data

        data_df = pd.DataFrame(box, index=range(0, 256), columns=range(0, 256))
        data_df = data_df.transpose()
        data_df = data_df.sort_index(ascending=False)
    
        image = data_df.to_numpy()
        image = np.reshape(image, (-1, 256, 256, 1))     
        predictions = model.predict(image).round(3)
        print(predictions, type(predictions))
     
        pred_labels = np.argmax(predictions, axis=1)
        print(int(pred_labels))
        print( class_names[int(pred_labels)])

        
        df = pd.DataFrame({
            'FileName': file_name,
            'Label': [class_names[int(pred_labels)]],
            'Corona': predictions[0,0],
            'Noise': predictions[0,1],
            'Surface': predictions[0,2],
            'Void': predictions[0,3]
        })

        if not os.path.exists(date_time+'.csv'):
            df.to_csv(date_time + ".csv", mode="w", index=False, encoding="utf-8-sig")
        else:
            df.to_csv(date_time + ".csv", mode="a", index=False, header=False, encoding="utf-8-sig")
        
    except Exception as e:
        print("------------파일 변환 중 오류 발생-----------\n", e)
        print(traceback.format_exc())

    
class Target:
    watchDir = "./data/"
    #watchDir에 감시하려는 디렉토리를 명시한다.

    def __init__(self):
        self.observer = Observer()   #observer객체를 만듦

    def run(self):
        BaseManager.register('ModelSingleton', ModelSingleton)
        manager = BaseManager()
        manager.start()
        inst = manager.ModelSingleton()
        
        date_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        event_handler = Handler(inst, date_time)
        self.observer.schedule(event_handler, self.watchDir, recursive=True)
        self.observer.start()
        try:
            while True:
    
                time.sleep(1)
        except:
            self.observer.stop()
            print("Error")
            self.observer.join()

def DeleteAllFiles(filePath):
    try:
        if os.path.exists(filePath):
            for file in os.scandir(filePath):
                os.remove(file.path)
                
            return "Remove All File"
    except Exception as e:
        print("Directory Not Found")

'''
https://stackoverflow.com/questions/3671666/sharing-a-complex-object-between-processes
'''

class Handler(FileSystemEventHandler):
#FileSystemEventHandler 클래스를 상속받음.
#아래 핸들러들을 오버라이드 함
    #파일, 디렉터리가 move 되거나 rename 되면 실행
    def __init__(self, model, date_time) :
        self.model = model
        self.date_time = date_time
    
    
    def on_moved(self, event):
        print(event)

    def on_created(self, event): #파일, 디렉터리가 생성되면 실행
        print(event)
        # print("\nTarget============= ",self.model, " ===============================================\n")
        p = Process(target=prps2prpd, args=(1, event.src_path, self.model, self.date_time))
        p.start()
        p.join()

    def on_deleted(self, event): #파일, 디렉터리가 삭제되면 실행
        print(event)

    # def on_modified(self, event): #파일, 디렉터리가 수정되면 실행
    #     print(event)


class_names = ['Corona', 'Noise', 'Surface', 'Void']

if __name__ == '__main__': #본 파일에서 실행될 때만 실행되도록 함
    
    DeleteAllFiles("./data")

    w = Target()
    w.run()
    
   

    
   