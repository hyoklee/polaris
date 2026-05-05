#!/bin/bash
# miv_polaris_build.sh — Build neuroh5 + miv-simulator on Polaris login node
# Run on login node (has internet).  Compute nodes use the resulting venv on Lustre.
#
# Key learnings vs naive approach:
#   - h5py: install serial wheel (not parallel build); parallel build fails due to
#     numpy.distutils/msvccompiler interaction with newer setuptools
#   - mpi4py: use cray-python 3.1.4 (Cray-compiled) via --system-site-packages;
#     the manylinux wheel 4.1.1 has MPI_Comm ABI mismatch with neuroh5.io
#   - neuroh5 cmake: use gcc/g++ not nvc++ (-Wno-unknown-pragmas unsupported by nvc++)
#   - Python_EXECUTABLE (capital P) required for cmake FindPython3
#   - neuroh5 cmake must target cray-python mpi4py include dir, not venv mpi4py
#
# Usage:  bash ~/bin/miv_polaris_build.sh [--rebuild]

set -euo pipefail

REBUILD=${1:-}
WORKDIR=/lus/grand/projects/gpu_hack/iowarp/miv_polaris
VENV=$WORKDIR/venv
BUILD=$WORKDIR/neuroh5_build
NEUROH5_SRC=/home/hyoklee/neuroh5
MIV_SRC=/home/hyoklee/MiV-Simulator

# Use GCC-variant HDF5 and MPICH to match mpi4py 3.1.4 (libmpi_gnu_123.so)
# mpi4py from cray-python links against libmpi_gnu_123.so (GCC variant)
# neuroh5/io.so must link the SAME library to share MPI_Init state
GNU_HDF5=/opt/cray/pe/hdf5-parallel/1.14.3.7/gnu/12.3
GNU_MPICH=/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3
CRAY_PYTHON=/opt/cray/pe/python/3.11.7/bin/python3
CRAY_MPI4PY_INC=/opt/cray/pe/python/3.11.7/lib/python3.11/site-packages/mpi4py/include

mkdir -p $WORKDIR $BUILD

module load cray-hdf5-parallel

echo "===  Step 1: virtual env  ==="
if [ "$REBUILD" = "--rebuild" ] && [ -d $VENV ]; then
    rm -rf $VENV
fi
if [ ! -d $VENV ]; then
    # --system-site-packages inherits cray-python's mpi4py 3.1.4 + numpy 1.24.4
    $CRAY_PYTHON -m venv --system-site-packages $VENV
fi
source $VENV/bin/activate

echo "===  Step 2: pip — core deps  ==="
pip install --quiet --upgrade pip setuptools wheel "cython>=3.0"
# Serial h5py from wheel — parallel build fails (numpy.distutils/msvccompiler issue)
pip install --quiet "h5py>=3.0"
# If h5py pulled in mpi4py>=4.0 wheel, remove it to restore cray-python's 3.1.4
# (manylinux mpi4py 4.1.1 has MPI_Comm ABI mismatch with Cray MPICH at runtime)
pip uninstall -y mpi4py 2>/dev/null || true
python3 -c "import mpi4py; print('mpi4py', mpi4py.__version__, '(cray-compiled)')"
pip install --quiet \
    "scipy>=1.13" \
    "pyyaml>=6.0.2" \
    "pydantic>=2.8.2" \
    "networkx>=3.2.1" \
    "matplotlib>=3.9.2" \
    "quantities>=0.16.0" \
    "click>=8.0"

echo "===  Step 3: NEURON  ==="
pip install --quiet "neuron>=8.2.6"
python3 -c "import neuron; print('NEURON', neuron.__version__)" 2>/dev/null || true

echo "===  Step 4: optional deps  ==="
pip install --quiet "machinable>=4.10.3" || echo "WARN: machinable failed"
pip install --quiet "treverhines-rbf @ git+https://github.com/treverhines/RBF" || echo "WARN: rbf failed"
pip install --quiet "spike_encoder @ git+https://github.com/iraikov/neural_spike_encoding.git" || echo "WARN: spike_encoder failed"

echo "===  Step 5: neuroh5 cmake build  ==="
cd $BUILD
if [ "$REBUILD" = "--rebuild" ]; then
    rm -f $BUILD/CMakeCache.txt
fi
if [ ! -f $BUILD/CMakeCache.txt ]; then
    cmake $NEUROH5_SRC \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DHDF5_ROOT=$GNU_HDF5 \
        -DHDF5_DIR=$GNU_HDF5 \
        "-DHDF5_LIBRARIES=$GNU_HDF5/lib/libhdf5.so;$GNU_HDF5/lib/libhdf5_hl.so" \
        -DMPI_C_COMPILER=$GNU_MPICH/bin/mpicc \
        -DMPI_CXX_COMPILER=$GNU_MPICH/bin/mpicxx \
        -DPython_EXECUTABLE=$VENV/bin/python3 \
        -DPython_ROOT_DIR=$VENV \
        -DMPI4PY_INCLUDE_DIR=$CRAY_MPI4PY_INC \
        -DBUILD_TESTS=OFF \
        2>&1 | tee $WORKDIR/neuroh5_cmake.log
fi
make -j8 python_neuroh5_io 2>&1 | tee -a $WORKDIR/neuroh5_cmake.log | tail -5

echo "===  Step 6: install neuroh5 Python package  ==="
NEUROH5_PKG=$VENV/lib/python3.11/site-packages/neuroh5
mkdir -p $NEUROH5_PKG
cp $BUILD/lib/io.so $NEUROH5_PKG/
cp -r $NEUROH5_SRC/python/neuroh5/. $NEUROH5_PKG/
python3 -c "import neuroh5; from neuroh5.io import scatter_read_trees; print('neuroh5 import OK')"

echo "===  Step 7: install miv-simulator  ==="
cd $MIV_SRC
pip install --quiet --no-deps -e .
python3 -c "import miv_simulator; print('miv_simulator import OK')"

echo ""
echo "Build complete.  Venv: $VENV"
echo "Submit test job:  qsub ~/bin/miv_polaris_test.pbs"
