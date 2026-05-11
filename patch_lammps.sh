#!/bin/bash
# patch_lammps.sh [-e] /path/to/lammps/

do_e_mode=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    -e)
      do_e_mode=true
      shift
      ;;
    -h|--help)
      echo "patch_lammps.sh [-e] /path/to/lammps/"
      exit
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1"
      echo "patch_lammps.sh [-e] /path/to/lammps/"
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

# https://stackoverflow.com/a/9472919
lammps_dir=$1

if [ "$lammps_dir" = "" ];
then
    echo "lammps_dir must be provided"
    exit 1
fi

if [ ! -d "$lammps_dir" ]
then
    echo "$lammps_dir doesn't exist"
    exit 1
fi

if [ ! -d "$lammps_dir/cmake" ]
then
    echo "$lammps_dir doesn't look like a LAMMPS source directory"
    exit 1
fi

# Check and produce nice message
if [ ! -f pair_nequip_allegro.cpp ]; then
    echo "Please run 'patch_lammps.sh' from the 'pair_nequip_allegro' root directory."
    exit 1
fi

# Check for double-patch
if grep -q "find_package(Torch REQUIRED)" $lammps_dir/cmake/CMakeLists.txt ; then
    echo "This LAMMPS installation _seems_ to already have been patched; please check it!"
    # exit 1
fi

if [ "$do_e_mode" = true ]
then
    echo "Making source symlinks (-e)..."
    for file in *_allegro.*; do
        ln -sf `realpath -s $file` $lammps_dir/src/$file
    done
    for file in ./fix/fix_addbornforce.*; do
        ln -sf `realpath -s $file` $lammps_dir/src/$(basename $file)
    done
    for file in ./compute/compute_nequip_allegro.*; do
        ln -sf `realpath -s $file` $lammps_dir/src/$(basename $file)
    done
    for file in *_allegro_kokkos.*; do
        ln -sf `realpath -s $file` $lammps_dir/src/KOKKOS/$file
    done
else
    echo "Copying files..."
    for file in *_allegro.*; do
        cp $file $lammps_dir/src/$file
    done
    for file in ./fix/fix_addbornforce.*; do
        cp $file $lammps_dir/src/$(basename $file)
    done
    for file in ./compute/compute_nequip_allegro.*; do
        cp $file $lammps_dir/src/$(basename $file)
    done
    for file in *_allegro_kokkos.*; do
        cp $file $lammps_dir/src/KOKKOS/$file
    done
fi

echo "Updating CMakeLists.txt..."

sed -i "s/set(CMAKE_CXX_STANDARD 11)/set(CMAKE_CXX_STANDARD 17)/" $lammps_dir/cmake/CMakeLists.txt

# Add libtorch
cat >> $lammps_dir/cmake/CMakeLists.txt << "EOF2"

message(STATUS "<< NEQUIP flags >>")
if(NEQUIP_AOT_COMPILE)
  add_compile_definitions(NEQUIP_AOT_COMPILE)
  message(STATUS "NEQUIP_AOT_COMPILE is enabled.")
else()
  message(STATUS "NEQUIP_AOT_COMPILE is disabled.")
endif()
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
EOF2

echo "Done!"
