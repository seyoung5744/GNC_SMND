import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

prpd = pd.read_csv("./SMND_345kV_EBG_A_S_0A_62_20190515113600.dat[PRPD변환].csv", names=range(0,256)) # surface

prpd_numpy = prpd.to_numpy()
prpd_numpy = np.array(prpd_numpy, dtype=np.float32)
prpd_numpy[np.where(prpd_numpy== 0) ] = np.nan

# custom cmap => https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
color_num = 10
viridis = plt.cm.get_cmap('YlOrRd', color_num)
newcolors = viridis(np.linspace(0, 1, color_num))

newcolors[-1:, :] = np.array([1/256, 1/256, 1/256, 1]) # black

newcmp = ListedColormap(np.flipud(newcolors))

plt.figure(figsize=(10,10))


plt.imshow(prpd_numpy, cmap=newcmp)
# plt.clim(-10,np.max(prpd_numpy))
plt.colorbar()
plt.savefig("./test_image.jpg")
plt.show()