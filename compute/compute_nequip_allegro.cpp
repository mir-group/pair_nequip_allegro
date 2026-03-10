/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#include "compute_nequip_allegro.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "pair_nequip_allegro.h"
#include "update.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>

using namespace LAMMPS_NS;

template <bool nequip_mode, int peratom>
ComputeNequIPAllegro<nequip_mode, peratom>::ComputeNequIPAllegro(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{
  if (nequip_mode)
  {
    compute_name = "nequip";
  }
  else
  {
    compute_name = "allegro";
  }

  if constexpr (!peratom)
  {
    // compute 1 all allegro quantity length
    if (narg != 5)
      error->all(FLERR, "Incorrect args for compute {}", compute_name);
  }
  else
  {
    // compute 1 all allegro/atom quantity length newton(1/0)
    if (narg != 6)
      error->all(FLERR, "Incorrect args for compute {}/atom", compute_name);
  }

  if (strcmp(arg[1], "all") != 0)
    error->all(FLERR, "compute {} can only operate on group 'all'", compute_name);

  quantity = arg[3];
  if constexpr (peratom)
  {
    peratom_flag = 1;
    nperatom = std::atoi(arg[4]);
    newton = std::atoi(arg[5]);
    if (newton)
    {
      comm_reverse = nperatom;
      if (nequip_mode)
      {
        error->all(FLERR, "compute {} cannot use newton reverse communication, please set to 0", compute_name);
      }
    }
    size_peratom_cols = nperatom == 1 ? 0 : nperatom;
    nmax = -12;
    if (comm->me == 0)
      utils::logmesg(lmp, "compute {}/atom will evaluate the quantity {} of length {} with newton {}\n", compute_name,
                     quantity, size_peratom_cols, newton);
  }
  else
  {
    vector_flag = 1;
    // As stated in the README, we assume vector properties are extensive
    extvector = 1;
    size_vector = std::atoi(arg[4]);
    if (size_vector <= 0)
      error->all(FLERR, "Incorrect vector length!");
    memory->create(vector, size_vector, "ComputeNequIPAllegro:vector");
    if (comm->me == 0)
      utils::logmesg(lmp, "compute {} will evaluate the quantity {} of length {}\n", compute_name,
                     quantity, size_vector);
  }

  assert_pair_compatibility();

  if (nequip_allegro_pair == nullptr)
  {
    error->all(FLERR, "no compatible pair style; compute {} must be defined after pair style", compute_name);
  }

  nequip_allegro_pair->add_custom_output(quantity);
}

template <bool nequip_mode, int peratom>
void ComputeNequIPAllegro<nequip_mode, peratom>::init()
{
  ;
}

template <bool nequip_mode, int peratom>
ComputeNequIPAllegro<nequip_mode, peratom>::~ComputeNequIPAllegro()
{
  if (copymode)
    return;
  if constexpr (peratom)
  {
    memory->destroy(vector_atom);
  }
  else
  {
    memory->destroy(vector);
  }
}

template <bool nequip_mode, int peratom>
void ComputeNequIPAllegro<nequip_mode, peratom>::compute_vector()
{
  invoked_vector = update->ntimestep;

  // empty domain, pair style won't store tensor
  // note: assumes nlocal == inum
  if (atom->nlocal == 0)
  {
    for (int i = 0; i < size_vector; i++)
    {
      vector[i] = 0.0;
    }
  }
  else
  {
    const torch::Tensor &quantity_tensor =
        ((PairNequIPAllegro<nequip_mode> *)force->pair)->custom_output.at(quantity).cpu().ravel();

    auto quantity = quantity_tensor.data_ptr<double>();

    if (quantity_tensor.size(0) != size_vector)
    {
      error->one(FLERR, "size {} of quantity tensor {} does not match expected {} on rank {}",
                 quantity_tensor.size(0), this->quantity, size_vector, comm->me);
    }

    for (int i = 0; i < size_vector; i++)
    {
      vector[i] = quantity[i];
    }
  }

  // even if empty domain
  MPI_Allreduce(MPI_IN_PLACE, vector, size_vector, MPI_DOUBLE, MPI_SUM, world);
}

template <bool nequip_mode, int peratom>
void ComputeNequIPAllegro<nequip_mode, peratom>::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  if (atom->nmax > nmax)
  {
    nmax = atom->nmax;
    memory->destroy(array_atom);
    memory->create(array_atom, nmax, nperatom, "ComputeNequIPAllegroPerAtom:array");
    for (int i = 0; i < nmax; i++)
    {
      for (int j = 0; j < nperatom; j++)
      {
        array_atom[i][j] = 0.0;
      }
    }
    if (nperatom == 1)
      vector_atom = &array_atom[0][0];
  }

  // guard against empty domain (pair style won't store tensor)
  if (atom->nlocal > 0)
  {
    const torch::Tensor &quantity_tensor =
        ((PairNequIPAllegro<nequip_mode> *)force->pair)->custom_output.at(quantity).cpu().contiguous().reshape({-1, nperatom});

    auto quantity = quantity_tensor.accessor<double, 2>();

    int nlocal = atom->nlocal;
    int ntotal = nlocal + atom->nghost;
    int nquantity = nequip_mode ? nlocal : ntotal; // Same as the pair style's logic for forces

    for (int i = 0; i < nquantity; i++)
    {
      for (int j = 0; j < nperatom; j++)
      {
        array_atom[i][j] = quantity[i][j];
      }
    }
  }

  // even if empty domain
  if (newton)
    comm->reverse_comm(this);
}

template <bool nequip_mode, int peratom>
int ComputeNequIPAllegro<nequip_mode, peratom>::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++)
  {
    for (int j = 0; j < nperatom; j++)
    {
      buf[m++] = array_atom[i][j];
    }
  }
  return m;
}

template <bool nequip_mode, int peratom>
void ComputeNequIPAllegro<nequip_mode, peratom>::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++)
  {
    j = list[i];
    for (int k = 0; k < nperatom; k++)
    {
      array_atom[j][k] += buf[m++];
    }
  }
}

template <bool nequip_mode, int peratom>
void ComputeNequIPAllegro<nequip_mode, peratom>::assert_pair_compatibility()
{
  Pair *pair = nullptr;

  if (nequip_mode)
  {
    pair = force->pair_match("nequip", 1);
  }
  else
  {
    pair = force->pair_match("allegro", 1);
    if (pair == nullptr)
      pair = force->pair_match("allegro/kk", 1);
  }

  nequip_allegro_pair = dynamic_cast<PairNequIPAllegro<nequip_mode> *>(pair);

  if (!nequip_allegro_pair)
  {
    error->all(FLERR, "Incompatible pair style for compute {}", compute_name);
  }
}

namespace LAMMPS_NS
{
  template class ComputeNequIPAllegro<false, 0>;
  template class ComputeNequIPAllegro<false, 1>;
  template class ComputeNequIPAllegro<true, 0>;
  template class ComputeNequIPAllegro<true, 1>;
}
