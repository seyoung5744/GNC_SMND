import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

prpd = pd.read_csv("./SMND_345kV_EBG_A_S_0A_62_20190515113600.dat[PRPD변환].csv", names=range(0,256)) # surface
# custom cmap => https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html

def prpd2image(prpd):
    prpd_numpy = prpd.to_numpy()
    prpd_numpy += 1
    prpd_numpy[np.where(prpd_numpy== 1) ] = 0

    color_num = np.max(prpd_numpy)
    viridis = plt.cm.get_cmap('YlOrRd', color_num)
    newcolors = viridis(np.linspace(0, 1, color_num))

    newcolors[-1:, :] = np.array([256/256, 256/256, 256/256, 1]) # white
    newcolors[-10:-1, :] = np.array([1/256, 1/256, 1/256, 1]) # black
    newcmp = ListedColormap(np.flipud(newcolors))

    plt.figure(figsize=(10,7))


    plt.imshow(prpd_numpy, cmap=newcmp)

    plt.colorbar()
    plt.clim(-1, np.max(prpd_numpy))
    plt.savefig("./test_image.jpg")
    plt.show()