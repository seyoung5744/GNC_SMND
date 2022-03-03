import os
import pandas as pd
import numpy as np

def get_threshold(prpd):
    """
    a. 2D Array의 각행을 Sum 한다. 2D -> 1D
    - out : 1D Array
    """
    row_sum_values = prpd.sum(axis=1)
        
        
    """
    b. 1D Array의 가장 큰값과 큰값이 속한 Index를 구한다.
    - out : max, max_index
    """
    max_value = np.max(row_sum_values)
    max_index = np.argmax(row_sum_values)

    """
    c. a의 Array 의 각 값에 b의 Max값을 나눈다. a / b.max
    - out : 1D Array
    """
    sums_devide_max = np.divide(row_sum_values, max_value)
        
    """
    d. c의 Array의 앞뒤 값의 차를 구한다. c.array(x-1) - c.array(x) 
    - 1D Array(length: 256) ⇒ 1D Array(length: 255)
    """
    current_subtract_post = np.diff(sums_devide_max)
        
    """
    e. d의 Array의 절대값 취한다. ABS(d)
    - out : 1D Array
    """
    current_subtract_post_to_abs = np.abs(current_subtract_post)
        
    """
    f. 1 - e.Array
    - out : 1D Array
    """
    one_subtract_abs = (1 - current_subtract_post_to_abs)
    
    """
    g. f. Array의 b의 max_index 부터 0.999보다 크거나 같은 Index 찾기
    - out : Index
        
    return search_index1
    """
    search_index1 = np.where(one_subtract_abs[max_index:] >= 0.999)[0][0] + max_index
    """
    h. f.Array의 g의 Index+1 로 부터 0.4995보다 크거나 같은 Index 찾기
    - out : Index
        
    return search_index2
    """
    start_index = search_index1 + 1
    search_index2 = np.where(one_subtract_abs[start_index :] >= 0.4995)[0][0] + start_index
        
    """
    i. f.Array의 g의 Index로 부터 20개의 Array 취득.
    - 20개의 Array중 0.9보다 작은 값이 있는가?
    - out : boorean
    """
    b = one_subtract_abs[search_index1: search_index1+20]

    isBoolean = np.isin(True, b < 0.9)
        
    """
    threshold 정하기
    """
    return search_index2 if isBoolean else search_index1

thresholds = {}

def get_threshold_dict():
    for root, subdirs, files in os.walk("./data/05. 표준데이터(PRPD)"):
        if len(root.split("\\")) != 1:
            sub_folder_name = root.split("\\")[1]
        else:
            continue
        print(sub_folder_name)

        file_list = os.listdir(root)

        threshold = {}
        for i in range(len(file_list)):
            prpd = pd.read_csv(root + "\\" + file_list[i], names=range(0, 256))


            # prpd 인덱스 뒤집기
            prpd = prpd.to_numpy()
            prpd = prpd[::-1]
            prpd = pd.DataFrame(prpd,columns=range(0,256))


            file_name = file_list[i].split('[')[0]

            th = get_threshold(prpd)

            """
            k. 해당 되는 노이즈 레벨 ( j ) 값 만큼 PRPD 2D의 0행부터 j행만큼  0으로 replace.
            """
            prpd.loc[: th] = 0


            """
            k로 인해 backgroud noise 가 남아있는지 확인을 하는데
            background noise하고 함은 k의 PRPD의 각 행의 1D Array가 모두 0보다 크다는 것은 1D Array모두 값이 채워져 있으면 a부터 다시 진행하라는 것
            """
            """
            l. k의 PRPD의 각 행의 1D Array가 모두 0보다 크면 k의 PRPD를 가지고 a부터 다시 1번만 반복.
            """
            new_sums = prpd.sum(axis=1)
            greater_than_zero = np.where(new_sums>0)[0]

            isBool = False

            # 0보다 큰 값을 가진 행의 길이가 256이면 모두 0보다 크다는 의미
            for i in range(len(greater_than_zero)):
                if len(np.where(prpd.loc[greater_than_zero[i]][:]>0)[0])==256: 
                    isBool = True

            # isBool에 따라 threshold를 다시 얻어올지...
            if isBool: # True
                print(file_name)
                threshold[file_name] = get_threshold(prpd)
            else: # False
                threshold[file_name] = th





        thresholds[sub_folder_name] = threshold
        
    return thresholds


convert_folder_dir = "./data/02. 표준데이터(PSA)2/"

def prps2psa():
    thresholds_dict = get_threshold_dict()
    print(thresholds_dict)
    for root, subdirs, files in os.walk("./data/01. 표준데이터(PRPS)"):
        if len(root.split("\\")) != 1:
            sub_folder_name = root.split("\\")[1] # 00. Void()
        else:
            continue
        
        #* 폴더 라벨링을 위한 split
        labeling = sub_folder_name.split(" ")[1] # Corona, Void...
        print("진행 중... ", labeling)
        
        added_folder_name = convert_folder_dir + labeling #* ./data/05. 표준데이터(PRPD)/Void ... ./data/05. 표준데이터(PRPD)/Corona
        
        
        if not os.path.exists(added_folder_name):
            os.makedirs(added_folder_name)

        file_list = os.listdir(root)
        
        thresholds = thresholds_dict[labeling]

        for i in range(len(file_list)):
            prps = pd.read_csv(root + "\\" + file_list[i], names=range(0, 256))
            
            
            """
            sin함수 x축 256칸으로 분리

            - 첫 번째 y 값 : 0
            - 마지막 y 값 : -2.44929360e-16
            """
            start = 0
            end = 2 * np.pi

            x = np.linspace(start, end, 256)
            len(np.sin(x))
            
            """
            prps 한 행에서 문턱값(Th)보다 큰 인덱스만 추출
            """
            file_name = file_list[i].split('[')[0]
            
            Th = thresholds[file_name] # 문턱값(Th)
        
            index_list = np.where(prps[:] > Th)[1]
            
            """
            Th를 적용한 인덱스에 해당하는 sin(x) 값
            """
            sin_values = np.sin(x)[index_list]
            
            """
            sin값들을 이용한 x 값 계산 : (현재 - 과거)
            """
            X = np.diff(sin_values)
            
            """
            sin값들을 이용한 y 값 계산 : (미래 - 과거)
            """
            Y = []

            for idx in range(1, len(sin_values)-1):
                Y.append(sin_values[idx+1] - sin_values[idx-1])
            
            """
            계산을 편의를 위해 np.array로 변환

            - list의 경우 [1,2,3] + 3 계산 불가
                - [1,2,3] + 3 (X)
            - np.array의 경우 [1,2,3] + 3 계산 가능
                - np.array([1,2,3]) + 3 => [4,5,6]
            """
            Y = np.array(Y)
            
            """
            PSA Mapping 좌표 변환
            - out : float
            """
            X_mapping = ((X + 2) /4) * 255
            Y_mapping = ((Y + 2) /4) * 255
            
            """
            소수 -> 정수
            """
            X_mapping = X_mapping[:-1]
            X_mapping = (X_mapping).astype(np.int)
            Y_mapping = (Y_mapping).astype(np.int)
                
            """
            X, Y 좌표에 따른 csv추출
            """
            block = np.zeros((256,256))

            for i in range(len(X_mapping)):
                block[X_mapping[i], Y_mapping[i]] += 1
                
            pd.DataFrame(block).to_csv(convert_folder_dir + labeling + "/" + file_name + "[PSA변환].csv"  , index = False, header=False,  mode="w")
            

prps2psa()