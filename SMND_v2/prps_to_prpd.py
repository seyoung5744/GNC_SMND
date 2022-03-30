import os
import pandas as pd
import numpy as np



sub_folder_name = ""
CONVERTED_FOLDER_DIR = "./data/05. 표준데이터(PRPD)/"
ORIGIN_FOLDER_DIR = "./data/01. 표준데이터(PRPS)/"

root_labels = [label for label in os.listdir(ORIGIN_FOLDER_DIR)] # ['03. Noise (54)', '00. Void (204)', '02. Surface (144)', '01. Corona (81)']

# Create save folder
for label in root_labels:
    added_folder_name = CONVERTED_FOLDER_DIR+label #* ./data/05. 표준데이터(PRPD)/Void ... ./data/05. 표준데이터(PRPD)/Corona
    if not os.path.exists(added_folder_name):
        os.makedirs(added_folder_name)
        
# Preprocessing prps to prpd    
for label in root_labels:
    file_list = os.listdir(ORIGIN_FOLDER_DIR + label)

    for i in range(len(file_list)):
        prps = pd.read_csv(ORIGIN_FOLDER_DIR + label + "/" + file_list[i], names=range(0,256))
        
        box = np.full((256,256), 0)
        for row, data_series in prps.iteritems():
            datas = data_series.value_counts()
            for col, data in datas.iteritems():
                box[row,col] = data


        data_df = pd.DataFrame(box, index = range(0, 256), columns=range(0, 256))
        data_df = data_df.T
        data_df = data_df.sort_index(ascending=False)
        
        data_df.to_csv(CONVERTED_FOLDER_DIR + label + "/" + file_list[i].split("[")[0] + "[PRPD변환].csv"  , index = False, header=False,  mode="w")