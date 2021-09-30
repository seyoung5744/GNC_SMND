import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

psa = pd.read_csv("./SMND_345kV_EBG_A_S_0A_62_20190515110800.dat[PSA변환].csv", names=range(0,256)) # surface
psa_numpy = psa.to_numpy(dtype=np.float32)
psa_numpy[np.where(psa_numpy== 0) ] = -1

# custom cmap => https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
color_num = 10
viridis = plt.cm.get_cmap('YlOrRd', color_num)
newcolors = viridis(np.linspace(0, 1, color_num))

newcolors[-1:, :] = np.array([256/256, 256/256, 256/256, 1]) # white
newcolors[-2:-1, :] = np.array([1/256, 1/256, 1/256, 1]) # black

newcmp = ListedColormap(np.flipud(newcolors))

plt.figure(figsize=(10,10))

plt.imshow(psa_numpy, cmap=newcmp)

# plt.imshow(psa)
plt.clim(np.min(psa_numpy),np.max(psa_numpy))
plt.colorbar()
plt.savefig("./test_image.jpg")
plt.show()