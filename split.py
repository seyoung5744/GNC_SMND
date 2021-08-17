# %%
import os
import splitfolders
# %%
base_path = os.getcwd()
print(base_path)
# %%
splitfolders.ratio("data/psa", output="psa", seed=1337, ratio=(0.8, 0.2))
# %%
