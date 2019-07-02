import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from meep import materials
from sinus import create_vertices
sizex = 1
sizey = 2
resolution = 150
pml_th = 30 / resolution
fullx = sizex + 2 * pml_th
fully = sizey + 2 * pml_th
cell = mp.Vector3(fullx, fully, 0)
mat = materials.Au
#####################
# run modes
# geom - show geometry instead of actual simulation
# saveref - save the reference file, needed for field enhancement calculations
####################
aux_mat = mp.Medium(epsilon=1)
geom = False
saveref = True
####################

####################
# geometry, load AFM profile
####################

metal_vert = create_vertices(ampl=0.1, periodicity=0.5,
                             thickness=0.04, resolution=resolution, sizex=fully, y=True)
####################
# source paraneters
####################

cfreq = 1.5
fwidth = 1.5
comp = mp.Hz
sources = [mp.Source(mp.GaussianSource(frequency=cfreq, fwidth=fwidth),
                     size=mp.Vector3(0, fully, 0),
                     component=comp,
                     center=mp.Vector3(-sizex / 2, 0))]

# Absorber on grating side because of field divergence at metal/pml interface
pml_layers = [mp.PML(pml_th, direction=mp.X), mp.Absorber(pml_th, direction=mp.Y)]

# empty cell for reference run
geometry = []
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution,
                    filename_prefix="data",
                    split_chunks_evenly=False)

# define monitors for further spectra calculation
mon_height = sizey
nfreq = 200
mon_skip = sizex / 20  # small skip from PML layers
refl_fr = mp.FluxRegion(center=mp.Vector3(-sizex / 2 +
                                          mon_skip, 0, 0), size=mp.Vector3(0, mon_height, 0))
refl = sim.add_flux(cfreq, fwidth, nfreq, refl_fr)

tran_fr = mp.FluxRegion(center=mp.Vector3(
    sizex / 2 - mon_skip, 0, 0), size=mp.Vector3(0, mon_height, 0))
tran = sim.add_flux(cfreq, fwidth, nfreq, tran_fr)
pt = mp.Vector3(sizex / 3, 0, 0)
print("Starting reference run")

if geom:
    sim.init_sim()
else:
    if saveref:
        sim.run(mp.to_appended("ref", mp.at_every(0.5 / (cfreq + fwidth * 0.5),
                                                    mp.output_efield_x, mp.output_efield_y, mp.output_efield_z)),
                until_after_sources=mp.stop_when_fields_decayed(10, comp, pt, 1e-2))
    else:
        sim.run(until_after_sources=mp.stop_when_fields_decayed(10, comp, pt, 1e-2))


# for normalization run, save flux fields data for reflection plane
straight_refl_data = sim.get_flux_data(refl)
# save incident power for transmission plane
straight_tran_flux = mp.get_fluxes(tran)
sim.reset_meep()

###################
# STAGE 2
###################

# now same calculation, but with structure

if geom:
    # overwrite mat to be correctly displayed
    # plot2D function does not handle correctly
    # dispersive materials
    mat = mp.Medium(epsilon=5)


geometry = [mp.Prism(metal_vert, height=100,
                     center=mp.Vector3(0, 0, 0),
                     axis=mp.Vector3(0, 0, 1),
                     material=mat)]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution,
                    filename_prefix="data",
                    split_chunks_evenly=False,)

# same monitors
refl = sim.add_flux(cfreq, fwidth, nfreq, refl_fr)
tran = sim.add_flux(cfreq, fwidth, nfreq, tran_fr)
sim.load_minus_flux_data(refl, straight_refl_data)
print("starting main run")

if geom:
    sim.plot2D()
    if mp.am_master():
        plt.show()
    exit()

sim.run(mp.to_appended("norm", mp.at_every(0.5 / (cfreq + fwidth * 0.5),
                                         mp.output_efield_x, mp.output_efield_y, mp.output_efield_z)),
        until_after_sources=mp.stop_when_fields_decayed(10, comp, pt, 1e-2))

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

if mp.am_master():
    plt.figure()
    plt.plot(wl, Rs, 'b-', label='reflectance')
    plt.plot(wl, Ts, 'r-', label='transmittance')
    plt.plot(wl, 1 - Rs - Ts, 'go-', label='loss')
    # plt.axis([5.0, 10.0, 0, 1])
    plt.xlabel("wavelength (μm)")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.savefig("spectra.svg")
    plt.show()
