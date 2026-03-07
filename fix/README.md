
### Fix

We provide an experimental "fix" that allows you to add first order corrections to the energies and forces of your simulation to model dynamics under an electric field. This is designed to be used with Allegro-Pol models trained using the (Allegro-Pol)[https://github.com/mir-group/allegro-pol/] extension package, and deployed using `nequip-compile ... --target pair_allegro_bc`. Below is example syntax with a time-dependent but spatially uniform electric field applied along the z axis.

```
variable PERIOD index 100000
variable efield equal 1e-2*1.5*cos((step/${PERIOD})*2*3.14)
fix born all addbornforce 0.0 0.0 v_efield
```

In the LAMMPS input file, the fix must be specified after defining the `allegro` pair style.

The three arguments are the values of the electric field along the x, y, and z axes.

The fix does not set the `thermo_energy = 1` flag; in order to include the fix's adjustment to the global energy you must set `fix modify [fix_id] energy yes` (See LAMMPS docs)[https://docs.lammps.org/Developer_notes.html#fix-contributions-to-instantaneous-energy-virial-and-cumulative-energy]. This energy adjustment can be accessed via `f_[fix_id]` (see LAMMPS docs)[https://docs.lammps.org/fix.html].
