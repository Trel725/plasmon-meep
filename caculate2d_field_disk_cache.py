import numpy as np
import sys
import h5py_cache
from joblib import Parallel, delayed
import argparse

parser = argparse.ArgumentParser(
    description="Perform FFT on time-domain EM data to get EM enhancement")
parser.add_argument("file", type=str, help="Field file to use")
parser.add_argument("reffile", type=str, help="Reference file to use")
parser.add_argument("savefile", type=str, help="File to save results")
parser.add_argument("-s", "--size", type=int, help="Batch size")
parser.add_argument("-f", "--freq", type=float, help="Central gaussian frequency")
parser.add_argument("-w", "--width", type=float, help="Width of gaussian frequency")


args = parser.parse_args()

cfreq = args.freq
freq_width = args.width
data = h5py_cache.File(args.file, "r", chunk_cache_mem_size=500 * 1024**2)
print("Opening {} as data file...".format(sys.argv[1]))
keys = list(data.keys())
data = [data[key] for key in keys]  # read all datasets in the hf5 file
print("Components: {}".format(keys))
print("Data {}".format(data[0].shape))


rfile = h5py_cache.File(args.reffile, "r", chunk_cache_mem_size=500 * 1024**2)
reference = [rfile[key] for key in keys]
print("Reference {}".format(reference[0].shape))

l = data[0].shape[2]  # length of temporal dimension
interp = 4  # optional zero padding of data to increase resilution, which is equivalent to interpolation
flen = (l * interp) // 2  # length of data in frequency
# domain is 2 times smaller due to FFT symmetry
print("Temporal length of data {}, length of freq domain {}".format(l, flen))
period = 0.5 / (cfreq + freq_width * 0.5)  # period of maximal frequncy component
fmax = 0.5 / period  # maxumal frequnce in spectrum (Nyquist-Shannon)
print("Freq max {}".format(fmax))
freqs = np.linspace(0, 1, flen) * fmax  # change linspace to arange 0...f_nyq


def get_power(data, nproc=4, paral=True):
    # convert chunk of data from time domain to
    # squared frequency domain (power spectrum)
    sh = data.shape
    if paral:
        data = np.array_split(data, nproc, axis=1)
        fdata = Parallel(n_jobs=nproc, prefer="threads")(delayed(np.fft.fft)(d, flen * 2, axis=-1)
                                                         for d in data)
        fdata = np.concatenate(fdata, axis=1) / l
    else:
        fdata = np.fft.fft(data, flen * 2, axis=-1) / l
    fdata = 2 * np.abs(fdata[:, :, 0:flen])
    power = np.square(fdata)
    del(data)
    return power


filename = args.savefile

batch_size = 50
if args.size:
    batch_size = args.size

xlen = data[0].shape[0]
sh = data[0].shape
file = h5py_cache.File(filename, "w", chunk_cache_mem_size=500 * 1024**2)
dset = file.create_dataset("enhancement", (sh[0], sh[1], flen), dtype='f')
dset.attrs["freqs"] = freqs

#iterate over chunks, calculating power spectra and saving to file
for i in range(0, xlen, batch_size):
    print("Iteration {} from {}".format(i, xlen))
    sh = data[0][i:i + batch_size].shape
    d = np.zeros((sh[0], sh[1], flen))
    for j in data:
        print(j.name)
        d += get_power(j[i:i + batch_size])

    r = np.zeros((sh[0], sh[1], flen))
    for j in reference:
        print(j.name)
        r += get_power(j[i:i + batch_size])

    dset[i:i + batch_size, :, :] = d[:, :, :] / r[:, :, :]

file.close()
