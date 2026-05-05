#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:10:00
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -A gpu_hack
#PBS -N apptainer_iowarp_test
#PBS -o /home/hyoklee/apptainer_compute_test.log
#PBS -e /home/hyoklee/apptainer_compute_test.err
#PBS -j oe

set -e

APPTAINER_BIN="/soft/spack/base/0.8.1/install/linux-sles15-x86_64/gcc-12.3.0/apptainer-1.3.2-o4dxrioaegfbzftm2uzazrkn6tprrang/bin"
SQUASHFS_BIN="/soft/spack/base/0.8.1/install/linux-sles15-x86_64/gcc-12.3.0/squashfs-4.6.1-dzb5zj3rsvu3huut3snx344sqyllyrkn/bin"
export PATH="$APPTAINER_BIN:$SQUASHFS_BIN:$PATH"
SIF="/lus/grand/projects/gpu_hack/iowarp/deps-nvidia.sif"

echo "=== Apptainer Container Test on Polaris ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SIF: $SIF"
echo ""

# --- 1. Verify apptainer is available ---
echo "--- Apptainer version ---"
apptainer --version
echo ""

# --- 2. Test: exec (launch) ---
echo "--- Test: exec (launch container) ---"
apptainer exec "$SIF" echo "Container launched successfully" 2>&1
EXEC_STATUS=$?
if [ $EXEC_STATUS -eq 0 ]; then
    echo "exec: PASSED"
else
    echo "exec: FAILED (exit $EXEC_STATUS)"
fi
echo ""

# --- 3. Test: exec with OS info ---
echo "--- Test: OS info inside container ---"
apptainer exec "$SIF" cat /etc/os-release 2>&1 | head -5
echo ""

# --- 4. Test: exec with GPU (--nv flag) ---
echo "--- Test: GPU access inside container (--nv) ---"
apptainer exec --nv "$SIF" nvidia-smi 2>&1
NV_STATUS=$?
if [ $NV_STATUS -eq 0 ]; then
    echo "GPU (--nv): PASSED"
else
    echo "GPU (--nv): FAILED (exit $NV_STATUS)"
fi
echo ""

# --- 5. Test: instance start ---
echo "--- Test: instance start ---"
INSTANCE_NAME="iowarp-test-$$"
apptainer instance start "$SIF" "$INSTANCE_NAME" 2>&1
START_STATUS=$?
if [ $START_STATUS -eq 0 ]; then
    echo "instance start: PASSED"
else
    echo "instance start: FAILED (exit $START_STATUS)"
fi
echo ""

# --- 6. Test: instance list ---
echo "--- Test: instance list ---"
apptainer instance list 2>&1
echo ""

# --- 7. Test: exec into running instance ---
echo "--- Test: exec into running instance ---"
apptainer exec instance://"$INSTANCE_NAME" echo "Exec into instance succeeded" 2>&1
INST_EXEC_STATUS=$?
if [ $INST_EXEC_STATUS -eq 0 ]; then
    echo "instance exec: PASSED"
else
    echo "instance exec: FAILED (exit $INST_EXEC_STATUS)"
fi
echo ""

# --- 8. Test: instance stop (shutdown) ---
echo "--- Test: instance stop (shutdown) ---"
apptainer instance stop "$INSTANCE_NAME" 2>&1
STOP_STATUS=$?
if [ $STOP_STATUS -eq 0 ]; then
    echo "instance stop: PASSED"
else
    echo "instance stop: FAILED (exit $STOP_STATUS)"
fi
echo ""

# --- 9. Verify instance is gone ---
echo "--- Test: verify instance stopped ---"
apptainer instance list 2>&1
echo ""

echo "=== Test Complete ==="
echo "Summary:"
echo "  exec launch:    $([ $EXEC_STATUS -eq 0 ] && echo PASSED || echo FAILED)"
echo "  GPU (--nv):     $([ $NV_STATUS -eq 0 ] && echo PASSED || echo FAILED)"
echo "  instance start: $([ $START_STATUS -eq 0 ] && echo PASSED || echo FAILED)"
echo "  instance exec:  $([ $INST_EXEC_STATUS -eq 0 ] && echo PASSED || echo FAILED)"
echo "  instance stop:  $([ $STOP_STATUS -eq 0 ] && echo PASSED || echo FAILED)"
