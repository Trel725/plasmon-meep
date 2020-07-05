import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os
from multiviewer import multi_slice_viewer
parser = argparse.ArgumentParser(
    description="Visualize calculated field enhancement")
parser.add_argument("file", type=str, help="File to use")
parser.add_argument("-s", "--save", action="store_true",
                    help="Save calculated spectra to file")
parser.add_argument("-x", "--excel", action="store_true",
                    help="Save to excel insted of CSV")
parser.add_argument("-t", "--tskip", type=int, help="Number of pixels to skip")
args = parser.parse_args()
print("Opening {} as data file...".format(args.file))

file = h5py.File(args.file, "r")
keys = list(file.keys())
dset = [file[key] for key in keys]
dset = dset[0]  # should not be any more datasets
freqs = dset.attrs['freqs']
xlen = dset.shape

flen = freqs.shape[0]
half_flen = flen // 2  # Nyquist-Shannon
enhancement = np.zeros_like(freqs)
data = dset[:]
if args.tskip:  # skip N pixels from the sides to eliminate enhancement on boundaries
    skip_x = args.tskip
else:
    skip_x = 10
skip_y = 0
if args.tskip:
    skip_y = args.tskip
for i in range(flen):
    tmp_data = data[skip_x:-skip_x, skip_y:-skip_y, i]
    # taking 99.9 percentile instead of maximum to eliminate
    # hotspot enhancement
    enhancement[i] = np.percentile(tmp_data, 99.9)

skip_freq = np.argmin(np.abs(0.5 - freqs))
# skip the nonphysical low-frequncy part of spectra


if args.save:
    df = pd.DataFrame()
    df["lambda"] = 1000 / freqs[skip_freq:]
    df["enhancement"] = enhancement[skip_freq:]
    fname = os.path.basename(args.file)
    fname, _ = os.path.splitext(fname)
    if args.excel:
        df.to_excel("spectra_" + fname + ".xlsx", index=False)
    else:
        df.to_csv("spectra_" + fname + ".csv", index=False)
    exit()

plt.plot(1000 / freqs[skip_freq:], enhancement[skip_freq:])
plt.xlabel("Wavelength, nm")
plt.ylabel("$|\\vec{E}$/$\\vec{E_0}|^2$")
plt.grid()
plt.show()

# visualize data
multi_slice_viewer(data[skip_x:-skip_x, skip_y:-skip_y, :], index_function=lambda x: 1000 / freqs[x])

plt.show()
