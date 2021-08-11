# %%
import os
import splitfolders
# %%
base_path = os.getcwd()
print(base_path)
# %%
splitfolders.ratio("raw/tabular", output="data_tabular", seed=1337, ratio=(0.8, 0.2))
# %%
 