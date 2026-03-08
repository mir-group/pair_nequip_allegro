import pytest

import os
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import textwrap
from io import StringIO
from collections import Counter

import ase
import ase.build
import ase.io

import torch

from nequip.data import AtomicDataDict, from_ase, compute_neighborlist_
from nequip.nn import with_edge_vectors_


from conftest import (
    _check_and_print,
    LAMMPS,
    LAMMPS_ENV_PREFIX,
    HAS_KOKKOS,
    HAS_KOKKOS_CUDA,
    HAS_OPENMP,
    COMPILE_MODES,
)


# build valid combinations of kokkos, openmp, device
def _build_backend_combinations():
    combinations = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    for device in devices:
        # base case: no kokkos, no openmp
        combinations.append((False, False, device))

        # OpenMP case
        if HAS_OPENMP:
            combinations.append((False, True, device))

        # Kokkos case - but skip if KOKKOS_CUDA is enabled and device is CPU
        if HAS_KOKKOS:
            if not (HAS_KOKKOS_CUDA and device == "cpu"):
                combinations.append((True, False, device))

    return combinations

def _add_efield(structure: ase.Atoms,efield: np.ndarray):
    forces = structure.get_forces()
    potential_energy = structure.get_potential_energy()

    polarization = structure.calc.results["polarization"]
    polarizability = structure.calc.results["polarizability"]
    born_charges = structure.calc.results["born_effective_charges"]
    
    force_correction = np.einsum("ikj,k->ij",born_charges,efield)
    
    polarization_correction = np.einsum("...ij,...i->...j",polarizability,efield)
    polarization_withfield = polarization + polarization_correction
    
    energy_correction = -np.einsum("...i,...i->",polarization_withfield,efield)

    forces_withfield = forces + force_correction
    energy_withfield = potential_energy + energy_correction

    print("ASE forces",forces)
    print("ASE energy",potential_energy)
    print("ASE polarizability",polarizability)

    print("ASE force correction",force_correction)
    print("ASE energy correction",energy_correction)

    
    return energy_withfield,forces_withfield,polarization_withfield


@pytest.mark.parametrize(
    "kokkos,openmp,device",
    _build_backend_combinations(),
)
@pytest.mark.parametrize(
    "compile_mode",
    # i.e. torchscript or aotinductor
    list(COMPILE_MODES.keys()),
)
@pytest.mark.parametrize(
    "n_rank",
    [1, 2, 4],
)
def test_repro(
    deployed_allegro_pol_model,
    kokkos: bool,
    openmp: bool,
    device: str,
    compile_mode: str,
    n_rank: int,
):
    structure: ase.Atoms
    model_tmpdir, calc, structures, config, tol = deployed_allegro_pol_model
    model_file_path = model_tmpdir + f"/{device}_" + COMPILE_MODES[compile_mode]

    # decide which tests to use `n_rank` > 1
    if n_rank > 1:
        data_name = str(config["dataset_file_name"])
        r_max = float(config["cutoff_radius"])
        if not (
            any(fname in data_name for fname in ["CuPd-cubic-big.xyz", "Cu-cubic.xyz"])
            and r_max < 8.0
        ):
            pytest.skip(
                f"skipping `n_rank={n_rank}` Allegro test for {data_name} and `r_max={r_max}`"
            )

    num_types = len(config["chemical_symbols"])

    efield_x = 0.0
    efield_y = 0.0
    efield_z = 1e-2*1.5
    efield = np.array([efield_x,efield_y,efield_z])

    newline = "\n"
    periodic = all(structures[0].pbc)
    PRECISION_CONST: float = 1e6
    lmp_in = textwrap.dedent(
        f"""
        units		metal
        atom_style	atomic
        newton on
        thermo 1

        # get a box defined before pair_coeff
        {'boundary p p p' if periodic else 'boundary s s s'}

        read_data structure.data

        pair_style	allegro
        # note that ASE outputs lammps types in alphabetical order of chemical symbols
        # since we use chem symbols in this test, just put the same
        pair_coeff	* * {model_file_path} {' '.join(sorted(set(config["chemical_symbols"])))}
{newline.join('        mass  %i 1.0' % i for i in range(1, num_types + 1))}

        neighbor	1.0 bin
        neigh_modify    delay 0 every 1 check no

        fix		1 all nve

        timestep	0.001

        compute atomicenergies all pe/atom
        compute allegroatomicenergies all allegro/atom atomic_energy 1 0
        compute allegroforces all allegro/atom forces 3 1
        compute polarization all allegro polarization 3
        compute polarizability all allegro polarizability 9
        compute borncharges all allegro/atom born_effective_charges 9 1
        compute totalatomicenergy all reduce sum c_atomicenergies
        compute stress all pressure NULL virial  # NULL means without temperature contribution
        variable efieldx equal {efield_x}
        variable efieldy equal {efield_y}
        variable efieldz equal {efield_z}
        fix born all addbornforce v_efieldx v_efieldy v_efieldz

        thermo_style custom step time temp pe c_totalatomicenergy etotal press spcpu cpuremain c_stress[*] c_polarization[*] c_polarizability[*] f_born f_born[*]
        run 0
        print "$({PRECISION_CONST} * c_stress[1]) $({PRECISION_CONST} * c_stress[2]) $({PRECISION_CONST} * c_stress[3]) $({PRECISION_CONST} * c_stress[4]) $({PRECISION_CONST} * c_stress[5]) $({PRECISION_CONST} * c_stress[6])" file stress.dat
        print "$({PRECISION_CONST} * c_polarization[1]) $({PRECISION_CONST} * c_polarization[2]) $({PRECISION_CONST} * c_polarization[3])" file polarization.dat
        print "$({PRECISION_CONST} * c_polarizability[1]) $({PRECISION_CONST} * c_polarizability[2]) $({PRECISION_CONST} * c_polarizability[3]) $({PRECISION_CONST} * c_polarizability[4]) $({PRECISION_CONST} * c_polarizability[5]) $({PRECISION_CONST} * c_polarizability[6]) $({PRECISION_CONST} * c_polarizability[7]) $({PRECISION_CONST} * c_polarizability[8]) $({PRECISION_CONST} * c_polarizability[9])" file polarizability.dat
        print $({PRECISION_CONST} * pe) file pe.dat
        print $({PRECISION_CONST} * c_totalatomicenergy) file totalatomicenergy.dat
        print $({PRECISION_CONST} * f_born) file addbornforceenergy.dat
        print "$({PRECISION_CONST} * f_born[1]) $({PRECISION_CONST} * f_born[2]) $({PRECISION_CONST} * f_born[3])" file addbornforcepolarization.dat
        write_dump all custom output.dump id type x y z fx fy fz c_atomicenergies c_allegroatomicenergies c_allegroforces[*] c_borncharges[*] modify format float %20.15g
        """
    )

    # for each model,structure pair
    # build a LAMMPS input using that structure
    with tempfile.TemporaryDirectory() as tmpdir:
        # save out the LAMMPS input:
        infile_path = tmpdir + "/test_repro.in"
        with open(infile_path, "w") as f:
            f.write(lmp_in)
        # environment variables
        env = dict(os.environ)
        env["_NEQUIP_LOG_LEVEL"] = "DEBUG"
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""

        # save out the structure
        for structure in structures:
            ase.io.write(
                tmpdir + "/structure.data",
                structure,
                format="lammps-data",
            )

            # run LAMMPS
            OMP_NUM_THREADS = 2  # just some choice
            retcode = subprocess.run(
                " ".join(
                    # Allow user to specify prefix to set up environment before mpirun. For example,
                    # using `LAMMPS_ENV_PREFIX="conda run -n whatever"` to run LAMMPS in a different
                    # conda environment.
                    [LAMMPS_ENV_PREFIX]
                    +
                    # MPI options if MPI
                    # --oversubscribe necessary for GitHub Actions since it only gives 2 slots
                    # > Alternatively, you can use the --oversubscribe option to ignore the
                    # > number of available slots when deciding the number of processes to
                    # > launch.
                    ["mpirun", "--oversubscribe", "-np", str(n_rank), LAMMPS]
                    # Kokkos options if Kokkos
                    + (
                        [
                            "-sf",
                            "kk",
                            "-k",
                            "on",
                            ("g" if HAS_KOKKOS_CUDA else "t"),
                            str(
                                max(torch.cuda.device_count() // n_rank, 1)
                                if HAS_KOKKOS_CUDA
                                else OMP_NUM_THREADS
                            ),
                            "-pk",
                            "kokkos newton on neigh half",
                        ]
                        if kokkos
                        else []
                    )
                    # OpenMP options if openmp
                    + (
                        ["-sf", "omp", "-pk", "omp", str(OMP_NUM_THREADS)]
                        if openmp
                        else []
                    )
                    # input
                    + ["-in", infile_path]
                ),
                cwd=tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )

            # uncomment to view LAMMPS output
            _check_and_print(retcode)

            # Check the inputs:
            if n_rank == 1:
                # this will only make sense with one rank
                # load debug data:
                mi = None
                lammps_stdout = iter(retcode.stdout.decode("utf-8").splitlines())
                line = next(lammps_stdout, None)
                while line is not None:
                    if line.startswith("Allegro edges: i j rij"):
                        edges = []
                        while not line.startswith("end Allegro edges"):
                            line = next(lammps_stdout)
                            edges.append(line)
                        edges = np.loadtxt(StringIO("\n".join(edges[:-1])))
                        mi = edges
                        break
                    line = next(lammps_stdout)
                mi = {
                    "i": mi[:, 0:1].astype(int),
                    "j": mi[:, 1:2].astype(int),
                    "rij": mi[:, 2:],
                }

                # first, check the model INPUTS
                structure_data = from_ase(structure)
                structure_data = compute_neighborlist_(
                    structure_data, r_max=float(config.cutoff_radius)
                )
                structure_data = with_edge_vectors_(structure_data, with_lengths=True)
                lammps_edge_tuples = [
                    tuple(e)
                    for e in np.hstack(
                        (
                            mi["i"],
                            mi["j"],
                        )
                    )
                ]
                nq_edge_tuples = [
                    tuple(e.tolist())
                    for e in structure_data[AtomicDataDict.EDGE_INDEX_KEY].t()
                ]
                # same num edges
                assert len(lammps_edge_tuples) == len(nq_edge_tuples)
                if kokkos:
                    # In the kokkos version, the atom ij are not tags
                    # so the order can't be compared to nequip
                    # so we just check overall set quantities instead
                    # this is slightly less stringent but should still catch problems
                    # check counters of per-atom num edges are same
                    assert Counter(
                        np.bincount(mi["i"].reshape(-1)).tolist()
                    ) == Counter(
                        torch.bincount(
                            structure_data[AtomicDataDict.EDGE_INDEX_KEY][0]
                        ).tolist()
                    )
                    # check OVERALL "set" of pairwise distance is good
                    nq_rij = (
                        structure_data[AtomicDataDict.EDGE_LENGTH_KEY]
                        .reshape(-1)
                        .clone()
                        .numpy()
                    )
                    nq_rij.sort()
                    lammps_rij = mi["rij"].reshape(-1).copy()
                    lammps_rij.sort()
                    maxabs = np.max(np.abs(nq_rij - lammps_rij))
                    assert np.allclose(nq_rij, lammps_rij), f"MaxAbs: {maxabs:.8f}"
                else:
                    # check same number of i,j edges across both
                    assert Counter(e[:2] for e in lammps_edge_tuples) == Counter(
                        e[:2] for e in nq_edge_tuples
                    )
                    # finally, check for each ij whether the the "sets" of edge lengths match
                    nq_ijr = np.core.records.fromarrays(
                        (
                            structure_data[AtomicDataDict.EDGE_INDEX_KEY][0],
                            structure_data[AtomicDataDict.EDGE_INDEX_KEY][1],
                            structure_data[AtomicDataDict.EDGE_LENGTH_KEY].reshape(-1),
                        ),
                        names="i,j,rij",
                    )
                    # we can do "set" comparisons by sorting into groups by ij,
                    # and then sorting the rij _within_ each ij pair---
                    # this is what `order` does for us with the record array
                    nq_ijr.sort(order=["i", "j", "rij"])
                    lammps_ijr = np.core.records.fromarrays(
                        (
                            mi["i"].reshape(-1),
                            mi["j"].reshape(-1),
                            mi["rij"].reshape(-1),
                        ),
                        names="i,j,rij",
                    )
                    lammps_ijr.sort(order=["i", "j", "rij"])
                    assert np.allclose(nq_ijr["rij"], lammps_ijr["rij"])

            # load dumped data
            lammps_result = ase.io.read(
                tmpdir + "/output.dump", format="lammps-dump-text"
            )

            # --- now check the OUTPUTS ---
            structure.calc = calc

            # check output atomic quantities
            # These energy and forces are the forces from the model adjusted by the addbornforce fix and should match ASE with Efield adjustment.
            
            ase_energy_withfield, ase_forces_withfield,ase_polarization_withfield = _add_efield(structure,efield)
            max_force_err = np.max(
                np.abs(ase_forces_withfield - lammps_result.get_forces())
            )
            max_force_comp = np.max(np.abs(ase_forces_withfield))
            force_rms = np.sqrt(np.mean(np.square(ase_forces_withfield)))

            #Temporary 
            lammps_allegroforces = np.zeros_like(ase_forces_withfield)
            lammps_allegroforces[:,0] = lammps_result.arrays["c_allegroforces[1]"].reshape(-1)
            lammps_allegroforces[:,1] = lammps_result.arrays["c_allegroforces[2]"].reshape(-1)
            lammps_allegroforces[:,2] = lammps_result.arrays["c_allegroforces[3]"].reshape(-1)
            print("Lammps forces",lammps_result.get_forces())
            print("Raw lammps forces",lammps_allegroforces)

            np.testing.assert_allclose(
                ase_forces_withfield,
                lammps_result.get_forces(),
                atol=tol,
                rtol=tol,
                err_msg=f"Force max abs err: {max_force_err:.8g} (atol/rtol={tol:.3g}). Max force component: {max_force_comp}, Force RMS: {force_rms}",
            )

            lammps_potentialenergy = (
                np.fromstring(
                    Path(tmpdir + "/pe.dat").read_text(),
                    sep=" ",
                    dtype=np.float64,
                )
                / PRECISION_CONST
            )
            lammps_addbornforceenergy = (
                np.fromstring(
                    Path(tmpdir + "/addbornforceenergy.dat").read_text(),
                    sep=" ",
                    dtype=np.float64,
                )
                / PRECISION_CONST
            )
            np.testing.assert_allclose(
                ase_energy_withfield,
                lammps_potentialenergy + lammps_addbornforceenergy,
                atol=tol,
                rtol=tol,
                err_msg=f"Energy w/field err: {ase_energy_withfield - (lammps_potentialenergy + lammps_addbornforceenergy):.8g}.",
            )
            
            # These energies are the raw outputs from the model and should match ASE without Efield adjustment.
            np.testing.assert_allclose(
                structure.get_potential_energies(),
                lammps_result.arrays["c_atomicenergies"].reshape(-1),
                atol=tol,
                rtol=tol,
                err_msg=f"Max atomic energy error: {np.abs(structure.get_potential_energies() - lammps_result.arrays['c_atomicenergies'].reshape(-1)).max()}",
            )

            np.testing.assert_allclose(
                structure.get_potential_energies(),
                lammps_result.arrays["c_allegroatomicenergies"].reshape(-1),
                atol=tol,
                rtol=tol,
                err_msg=f"Max compute atomic energy error: {np.abs(structure.get_potential_energies() - lammps_result.arrays['c_allegroatomicenergies'].reshape(-1)).max()}",
            )

            
            # These forces are the raw outputs from the model and should match ASE without Efield adjustment.
            ase_forces = structure.get_forces()
            lammps_allegroforces = np.zeros_like(ase_forces)
            lammps_allegroforces[:,0] = lammps_result.arrays["c_allegroforces[1]"].reshape(-1)
            lammps_allegroforces[:,1] = lammps_result.arrays["c_allegroforces[2]"].reshape(-1)
            lammps_allegroforces[:,2] = lammps_result.arrays["c_allegroforces[3]"].reshape(-1)
            max_force_err = np.max(
                np.abs(structure.get_forces() - lammps_allegroforces)
            )
            max_force_comp = np.max(np.abs(structure.get_forces()))
            force_rms = np.sqrt(np.mean(np.square(structure.get_forces())))
            np.testing.assert_allclose(
                ase_forces,
                lammps_allegroforces,
                atol=tol,
                rtol=tol,
                err_msg=f"Max compute forces abs err: {max_force_err:.8g} (atol/rtol={tol:.3g}). Max force component: {max_force_comp}, Force RMS: {force_rms}",
            )

            # Polarization model outputs
            ase_polarization = structure.calc.results["polarization"]
            ase_polarization = np.array(ase_polarization, dtype=np.float64).reshape(-1)
            lammps_polarization = (
                np.fromstring(
                    Path(tmpdir + "/polarization.dat").read_text(),
                    sep=" ",
                    dtype=np.float64,
                )
                / PRECISION_CONST
            )
            np.testing.assert_allclose(
                ase_polarization,
                lammps_polarization,
                atol=tol,
                rtol=tol,
                err_msg=f"Polarization error: {np.abs(ase_polarization - lammps_polarization).max()}",
            )

            #We add the extrapolarization computed by the fix to the lammps polarization and compare with thease (model) polarization with correction.
            lammps_addbornforcepolarization = (
                np.fromstring(
                    Path(tmpdir + "/addbornforcepolarization.dat").read_text(),
                    sep=" ",
                    dtype=np.float64,
                )
                / PRECISION_CONST
            )
            np.testing.assert_allclose(
                ase_polarization_withfield,
                lammps_polarization + lammps_addbornforcepolarization,
                atol=tol,
                rtol=tol,
                err_msg=f"Polarization w/Efield error: {np.abs(ase_polarization_withfield - (lammps_polarization + lammps_addbornforcepolarization)).max()}",
            )

            ase_polarizability = structure.calc.results["polarizability"]
            ase_polarizability = np.array(ase_polarizability, dtype=np.float64).reshape(3, 3)
            lammps_polarizability = (
                np.fromstring(
                    Path(tmpdir + "/polarizability.dat").read_text(),
                    sep=" ",
                    dtype=np.float64,
                )
                / PRECISION_CONST
            ).reshape(3, 3)
            np.testing.assert_allclose(
                ase_polarizability,
                lammps_polarizability,
                atol=tol,
                rtol=tol,
                err_msg=f"Polarizability error: {np.abs(ase_polarizability - lammps_polarizability).max()}",
            )

            ase_born_charges = structure.calc.results["born_effective_charges"]
            ase_born_charges = np.array(ase_born_charges, dtype=np.float64).reshape(-1, 9)
            lammps_born_charges = np.stack(
                [
                    lammps_result.arrays[f"c_borncharges[{i}]"].reshape(-1)
                    for i in range(1, 10)
                ],
                axis=1,
            )
            np.testing.assert_allclose(
                ase_born_charges,
                lammps_born_charges,
                atol=tol,
                rtol=tol,
                err_msg=f"Born charge error: {np.abs(ase_born_charges - lammps_born_charges).max()}",
            )

            # check system quantities
            lammps_pe = float(Path(tmpdir + "/pe.dat").read_text()) / PRECISION_CONST
            lammps_totalatomicenergy = (
                float(Path(tmpdir + "/totalatomicenergy.dat").read_text())
                / PRECISION_CONST
            )
            np.testing.assert_allclose(lammps_pe, lammps_totalatomicenergy)
            np.testing.assert_allclose(
                structure.get_potential_energy(), lammps_pe, atol=tol, rtol=tol
            )
            # in `metal` units, pressure/stress has units bars
            # so need to convert
            lammps_stress = np.fromstring(
                Path(tmpdir + "/stress.dat").read_text(), sep=" ", dtype=np.float64
            ) * (ase.units.bar / PRECISION_CONST)
            # https://docs.lammps.org/compute_pressure.html
            # > The ordering of values in the symmetric pressure tensor is as follows: pxx, pyy, pzz, pxy, pxz, pyz.
            lammps_stress = np.array(
                [
                    [lammps_stress[0], lammps_stress[3], lammps_stress[4]],
                    [lammps_stress[3], lammps_stress[1], lammps_stress[5]],
                    [lammps_stress[4], lammps_stress[5], lammps_stress[2]],
                ]
            )
            if periodic:
                # In LAMMPS, the convention is that the stress tensor, and thus the pressure, is related to the virial
                # WITHOUT a sign change.  In `nequip`, we chose currently to follow the virial = -stress x volume
                # convention => stress = -1/V * virial.  ASE does not change the sign of the virial, so we have
                # to flip the sign from ASE for the comparison.
                ase_stress = -structure.get_stress(voigt=False)
                stress_err = np.max(np.abs(ase_stress - lammps_stress))
                stol = tol * 20
                np.testing.assert_allclose(
                    ase_stress,
                    lammps_stress,
                    atol=stol,
                    rtol=stol,
                    err_msg=f"Stress max abs err: {stress_err:.8g} (tol={stol:.3g})\nASE stress: {ase_stress.flatten().tolist()}\nLAMMPS stress: {lammps_stress.flatten().tolist()}",
                )
