## What is it? 
**meep-plasmon** is a set of scripts for calculation of plasmon resonance/electric field enhancement on different structures.
Scripts are powered by FDTD solver MEEP, https://github.com/NanoComp/meep

## How it works?

1. Meep simulates interaction of structure of interest with EM waves, producing arrays describing field distribution in space and time.
Simulation is done twice, for the empty cell (reference run) and for the cell containing structure.
2. The FFT is performed on arrays from previous step, thus tranforming electric fields to frequency domain.
3. Complex values of tranformed fields are squared, producing energiy density at given frequncies. 
4. Densities of normal run are normalized by reference run, which directly gives EM field enhancement distribution
in space and frequency

## How to use it?

1. `mpirun -np 2 python ./perform_fdtd.py` perform step 1, parallel run on 2 processors.
The script includes sinusoidal grating as example, which could be easily modified by modification
of function definition in sinus.py or by defining own geometry.
2. `python  ./caculate2d_field_disk_cache.py -s 250 -f 1.5 -w 1.5 ./data-norm.h5 ./data-ref.h5 ./freqs.h5`
this script calculates FFT (parallel by default), its square and normalization.
3. `python ./visualize_freqs.py -t 50 ./freqs.h5` visualize the calculated field enhancement arrays.

## How it looks?

![](https://github.com/Trel725/plasmon-meep/raw/master/img/geometry.png)

## Notes

multiviewer.py is taken from https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
