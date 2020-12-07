import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


sizex = 10
sizey = 7
resolution = 50
pml_th = 35 / resolution
fullx = sizex + 2 * pml_th
fully = sizey + 2 * pml_th
cell = mp.Vector3(fullx, fully, 0)
mat = mp.Medium(epsilon=2)
supp_mat = mp.Medium(epsilon=1)
aux_mat = mp.Medium(epsilon=1)

# set to True to plot instead of run simulation
geom = False

###################
# X-compression ratio
x_comp = 2
###################

trf = mp.Matrix(diag=mp.Vector3(x_comp, 1, 1))
mat.transform(trf)
aux_mat.transform(trf)

slab_thickness = sizex / 10

geometry = [mp.Block(size=mp.Vector3(2 * x_comp * slab_thickness,
                                     fully, mp.inf),
                     center=mp.Vector3(0, 0, 0),
                     material=aux_mat)]

####################
# source params
cfreq = 1.25
fwidth = 1.5
comp = mp.Hz
#####################

sources = [mp.Source(mp.GaussianSource(frequency=cfreq, fwidth=fwidth),
                     size=mp.Vector3(0, sizey * 0.7, 0),
                     component=comp,
                     center=mp.Vector3(-sizex / 2, 0))]

pml_layers = [mp.PML(pml_th, direction=mp.X),
              mp.PML(pml_th, direction=mp.Y)]


sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution,
                    Courant=np.min([(0.5 / x_comp) / np.sqrt(2), 0.5]),
                    split_chunks_evenly=False)

mon_height = sizey * 0.75
nfreq = 200
mon_skip = sizex / 8

# monitor for reflected fields
refl_fr = mp.FluxRegion(center=mp.Vector3(-sizex / 2 + mon_skip, 0, 0),
                        size=mp.Vector3(0, mon_height, 0))
refl = sim.add_flux(cfreq, fwidth, nfreq, refl_fr)

# monitor for transmitted fields
tran_fr = mp.FluxRegion(center=mp.Vector3(sizex / 2 - mon_skip, 0, 0),
                        size=mp.Vector3(0, mon_height, 0))
tran = sim.add_flux(cfreq, fwidth, nfreq, tran_fr)

pt = mp.Vector3(sizex / 3, 0, 0)
print("starting reference run")

if not geom:
    sim.run(until_after_sources=mp.stop_when_fields_decayed(10, comp, pt, 1e-2))

# for normalization run, save flux fields data for reflection plane
straight_refl_data = sim.get_flux_data(refl)

# save incident power for transmission plane
straight_tran_flux = mp.get_fluxes(tran)

print("Resetting meep...")
sim.reset_meep()

block = mp.Block(size=mp.Vector3(x_comp * slab_thickness, sizey / 4, mp.inf),
                 center=mp.Vector3(0, 0, 0),
                 material=mat)

geometry.append(block)

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution,
                    Courant=np.min([(0.5 / x_comp) / np.sqrt(2), 0.5]),
                    split_chunks_evenly=False,
                    eps_averaging=True)

# again define monitors
refl = sim.add_flux(cfreq, fwidth, nfreq, refl_fr)
tran = sim.add_flux(cfreq, fwidth, nfreq, tran_fr)
sim.load_minus_flux_data(refl, straight_refl_data)

print("starting main run")

if geom:
    arr = sim.get_epsilon()
    sim.plot2D()
    if mp.am_master():
        plt.show()
    exit()

sim.run(mp.to_appended("fields", mp.at_every(0.5 / (cfreq + fwidth * 0.5),
                                             mp.output_efield_x,
                                             mp.output_efield_y,
                                             mp.output_efield_z)),
        until_after_sources=mp.stop_when_fields_decayed(10, comp, pt, 1e-3))


refl_flux = mp.get_fluxes(refl)
tran_flux = mp.get_fluxes(tran)

flux_freqs = mp.get_flux_freqs(refl)

wl = []
Rs = []
Ts = []
for i in range(nfreq):
    wl = np.append(wl, 1 / flux_freqs[i])
    Rs = np.append(Rs, -refl_flux[i] / straight_tran_flux[i])
    Ts = np.append(Ts, tran_flux[i] / straight_tran_flux[i])
    Abs = -np.log10(Ts)


if mp.am_master():
    res = pd.DataFrame([wl, Rs, Ts, Abs]).T
    res.to_csv("spectra_res{}_comp{}_stretched.csv".format(
        resolution, comp), index=False)
    plt.figure()
    plt.plot(wl, Rs, 'b-', label='reflectance')
    plt.plot(wl, Ts, 'r-', label='transmittance')
    plt.plot(wl, 1 - Rs - Ts, 'go-', label='loss')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig("spectra_stretched.svg")
    plt.show()
