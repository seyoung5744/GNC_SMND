import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def PD2Image(file_list):
    images = []
    for img_path in file_list:

        image = pd.read_csv(img_path, names=range(0, 256))
        image = np.pad(image, (2,2), 'constant', constant_values=0) # 256 -> 260
        image = np.reshape(image, (260, 260, 1))

        images.append(image)

    """
    channle 1D -> 3D 
    """        
    images = np.array(images)

    # 0 이외의 숫자 1로 정규화
    images[images > 0] = 1
    images = images.astype('float32')

    images_3 = np.full((images.shape[0], 260, 260, 3), 0.0)

    for i, s in enumerate(images):
        images_3[i] = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB) 
        
    """
    이미지 return (n, 260,260,3)
    """
    
    for image, file in zip(images_3, file_list):
        file_name = file_name.split('\\')[-1].split("csv")[0] # 파일 명
        plt.imsave(CONVERTED_JPEG_FOLDER_DIR + label + "/"+file_name+"jpeg",image) # 파일 
    