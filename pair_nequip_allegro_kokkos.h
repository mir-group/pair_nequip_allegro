/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(allegro/kk,PairAllegroKokkos<false>)

#else

#ifndef LMP_PAIR_ALLEGRO_KOKKOS_H
#define LMP_PAIR_ALLEGRO_KOKKOS_H

#include "pair_nequip_allegro.h"
#include <pair_kokkos.h>


namespace LAMMPS_NS {

template<bool nequip_mode>
class PairAllegroKokkos : public PairNequIPAllegro<nequip_mode> {
 public:
  typedef PairNequIPAllegro<nequip_mode> super;
  using DeviceType = LMPDeviceType;
  using MemberType = typename Kokkos::TeamPolicy<DeviceType>::member_type;
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef LMPDeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairAllegroKokkos(class LAMMPS *);
  virtual ~PairAllegroKokkos();
  virtual void compute(int, int);
  virtual void coeff(int, char **);
  virtual void init_style();

  KOKKOS_INLINE_FUNCTION
  void v_tally(KK_FLOAT (&v)[6], const int &i, const int &j,
      const KK_FLOAT &fx, const KK_FLOAT &fy, const KK_FLOAT &fz, const KK_FLOAT &delx,
                  const KK_FLOAT &dely, const KK_FLOAT &delz) const;

  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom; 
 protected:
  typedef Kokkos::DualView<int***,DeviceType> tdual_int_3d;
  typedef typename tdual_int_3d::t_dev_const_randomread t_int_3d_randomread;
  typedef typename tdual_int_3d::t_host t_host_int_3d;

  typename AT::t_kkfloat_1d_3_lr_randomread x; 
  typename AT::t_kkacc_1d_3 f; 
  typename AT::t_tagint_1d tag;
  typename AT::t_int_1d_randomread type;

  DAT::ttransform_kkacc_1d k_eatom; 
  DAT::ttransform_kkacc_1d_6 k_vatom;  

  using inputtype = typename super::inputtype;
  using outputtype = typename super::outputtype;

  using IntView1D = Kokkos::View<int*, Kokkos::LayoutRight, DeviceType>;
  using IntView2D = Kokkos::View<int**, Kokkos::LayoutRight, DeviceType>;
  using LongView1D = Kokkos::View<long*, Kokkos::LayoutRight, DeviceType>;
  using LongView2D = Kokkos::View<long**, Kokkos::LayoutRight, DeviceType>;
  using UnmanagedFloatView1D = Kokkos::View<outputtype*, Kokkos::LayoutRight, DeviceType>;
  using UnmanagedFloatView2D = Kokkos::View<outputtype**, Kokkos::LayoutRight, DeviceType>;
  using View1D = Kokkos::View<KK_ACC_FLOAT*, Kokkos::LayoutRight, DeviceType>;
  using View2D = Kokkos::View<KK_ACC_FLOAT**, Kokkos::LayoutRight, DeviceType>;
  using InputFloatView2D = Kokkos::View<inputtype**, Kokkos::LayoutRight, DeviceType>;



  IntView1D d_type_mapper;
  LongView1D d_ij2type;
  LongView2D d_edges;
  InputFloatView2D d_xfloat;

  View2D d_cutoff_matrix;



  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  //NeighListKokkos<DeviceType> k_list;

  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;

  Kokkos::View<int**,DeviceType> d_neighbors_short;
  Kokkos::View<int*,DeviceType> d_numneigh_short, d_cumsum_numneigh_short;


  friend void pair_virial_fdotr_compute<PairAllegroKokkos>(PairAllegroKokkos*);
};
}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot use chosen neighbor list style with pair_allegro/kk

Self-explanatory.

*/
