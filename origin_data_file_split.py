import os, shutil
 
sub_folder_name = ""

folder_list = ['01. 표준데이터(PRPS)', '02. 표준데이터(PSA)','03. 표준데이터(통계)','04. 표준데이터(.dat)']
filename = ['[PRPS변환]','[PSA변환]','[통계변환]','.dat']

for root, subdirs, files in os.walk("./data/00. 표준데이터_원본"):
    print(root)
    if len(root.split("\\")) != 1:
        sub_folder_name = root.split("\\")[1]

        for idx, f_list in enumerate(folder_list):

            if not os.path.exists(sub_folder_name):
                os.makedirs("./data/" + f_list + "/"+ sub_folder_name)

            print(sub_folder_name)
            for r,s,f in os.walk(root):
                for ff in f:
                    if filename[idx] in ff:
                        print(ff)
                        file_to_move = os.path.join(root, ff)
                        shutil.move(file_to_move, "./data/" + f_list+"/"+sub_folder_name)
                        # shutil.copy2(file_to_move, "./data/" + f_list+"/"+sub_folder_name)
     