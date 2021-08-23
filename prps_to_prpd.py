import os
import pandas as pd
import numpy as np



sub_folder_name = ""
convert_folder_dir = "./data/05. 표준데이터(PRPD)/"


for root, subdirs, files in os.walk("./data/01. 표준데이터(PRPS)"):
    print(root)
    if len(root.split("\\")) != 1:
        sub_folder_name = root.split("\\")[1]
    else:
        continue

    #* 폴더 라벨링을 위한 split
    labeling = sub_folder_name.split(" ")[1]

    added_folder_name = convert_folder_dir+labeling #* ./data/05. 표준데이터(PRPD)/Void ... ./data/05. 표준데이터(PRPD)/Corona
    if not os.path.exists(added_folder_name):
        os.makedirs(added_folder_name)
    

    print(labeling)
    file_list = os.listdir(root)
    # print(file_list)
    print(file_list[0])

    print(file_list[0].split("[")[0] + "[PRPD변환].csv")

    for i in range(len(file_list)):
        # print(root  + "\\" +file_list[i])

        prps = pd.read_csv(root + "\\" + file_list[i], names=range(0,256))

        box = np.full((256,256), 0)
        
        for j in range(0, len(prps)):
            for idx, data in enumerate(prps.loc[j]):
                box[idx][data] += 1

        data_df = pd.DataFrame(box, index = range(0, 256), columns=range(0, 256))
        data_df = data_df.transpose()
        data_df = data_df.sort_index(ascending=False)
        
        data_df.to_csv(convert_folder_dir + labeling + "/" + file_list[i].split("[")[0] + "[PRPD변환].csv"  , index = False, header=False,  mode="w")
    