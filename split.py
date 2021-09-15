# %%
import os
import splitfolders
# %%
base_path = os.getcwd()
print(base_path)

# folder_list = ['01. 표준데이터(PRPS)', '02. 표준데이터(PSA)','03. 표준데이터(통계)','04. 표준데이터(.dat)','05. 표준데이터(PRPD)']
folder_list = ['GADF','GASF',"RP"]
filename = ['[PRPS변환]','[PSA변환]','[통계변환]','.dat']


# splitfolders.ratio("data/01. 표준데이터(PRPS)", output="data/split_data/01. 표준데이터(PRPS)", seed=1337, ratio=(0.8,0.0, 0.2))

# for folder in folder_list:
#     print(folder)
#     splitfolders.ratio("data/" + folder, output="data/split_data/"+folder, seed=1337, ratio=(0.8,0.0, 0.2))
for folder in folder_list:
    print(folder)
    splitfolders.ratio("images/" + folder, output="images/split_data/"+folder, seed=1337, ratio=(0.8,0.0, 0.2))
# %%
