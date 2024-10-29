# %%
import os
import matplotlib.pyplot as plt
import numpy as np

# this is to see the prograss
from tqdm import tqdm
# this is for parallelism
from multiprocessing import Pool

# %%
def get_fig_aspect_ratio(xlen, ylen, base=5):
    """Get the aspect ratio to fit the data."""

    aspect_ratio = np.ceil(ylen / xlen)
    fx = base * (0.5 + aspect_ratio)
    fy = base

    return fx, fy

# %%
# Load the NPZ file
namehead = "tracer0_x0.0000"
field = "tracer0"
nplts = 11

f_max = np.zeros(nplts)
f_min = np.zeros(nplts)

for i in tqdm(range(nplts)):
    fdir = f"""{namehead}_{i}.npz"""
    data = np.load(fdir, allow_pickle=True)
    slices = data["slices"][()]

    f_max[i] = np.ceil(np.max(slices[field]))
    f_min[i] = np.floor(np.min(slices[field]))

for index in tqdm(range(nplts)):

    fdir = f"""{namehead}_{index}.npz"""
    data = np.load(fdir, allow_pickle=True)

    ds_attributes = data["ds_attributes"][()]
    slices = data["slices"][()]
    dxyz = ds_attributes["dxyz"]
    normal = data["normal"]
    iloc = data["iloc"]
    time = ds_attributes["time"]
    length_unit = ds_attributes["length_unit"]


    # %%
    # Inputs for plotting
    ylen, xlen = slices[field].shape
    fx, fy = get_fig_aspect_ratio(xlen, ylen, base=5)

    y = np.linspace(ds_attributes["left_edge"][1], ds_attributes["right_edge"][1], xlen)
    z = np.linspace(ds_attributes["left_edge"][2], ds_attributes["right_edge"][2], ylen)
    Y, Z = np.meshgrid(y, z, indexing="xy")

    # %%
    fig, ax = plt.subplots(1, 1, figsize=(fx, fy), dpi=300)
    im = ax.pcolormesh(Y,Z,slices[field])

    length_unit ="m"
    ax.set_xlabel(f"y ({length_unit})")
    ax.set_ylabel(f"z ({length_unit})")
    ax.set_title(f"""{normal} = {iloc:.2f} {length_unit}, """
    f"""time = {float(time.in_units("s")):.2f} s""")

    ax.set_box_aspect(1) 
    cbar = fig.colorbar(im, ax=ax)
    cbar.mappable.set_clim(np.min(f_min), np.max(f_max))

    # %% save figure
    res_dpi = 300
    imgpath = "./"

    fig.savefig(os.path.join(imgpath, f"""{field}_{normal}{iloc:.4f}_{index}.png"""),
                dpi=res_dpi,)



