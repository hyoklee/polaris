#!/bin/bash
# Congruence pipeline for ONE Terra Fusion granule, as ares' tf_all_granules.sh
# run_one did it, with the matched configuration of ares parts 11-12 (MODIS
# band 31, CERES WN_Radiance):
#   tf_aster_blocks.py -> aster_blocks.json
#   tf_sources.py      -> regrid_inputs_b31_WN.npz
#   tf_congruence.py   -> congruence_b31_WN.json   (regrid on GPU: TF_REGRID)
#   tf_contrast.py     -> contrast_b31_WN.json     (band-31 BT sd per block)
#
# Usage: tf_granule.sh <granule.h5> <work dir>
# Environment: PY (python), TF (this directory), PYTAF_DIR, TF_REGRID,
#   ZE_AFFINITY_MASK (the GPU tile this granule's regrid runs on).
# Writes <work dir>/run.log and stage_times.json; exits non-zero on failure.
set -uo pipefail
f=$1
w=$2
PY=${PY:-python3}
TF=${TF:-$(dirname "$0")}
export TF_GRANULE=$f MODIS_BAND_IDX=${MODIS_BAND_IDX:-10}
export CERES_FIELD=${CERES_FIELD:-WN_Radiance} TF_TAG=b31_WN
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}   # ares part 3: pytaf OpenMP only hurts
mkdir -p "$w" && cd "$w" || exit 1
exec >> run.log 2>&1
orbit=$(basename "$f" | sed -E 's/TERRA_BF_L1B_(O[0-9]+)_.*/\1/')
echo "=== $orbit $(basename "$f")  gpu=${CUDA_VISIBLE_DEVICES:-${ZE_AFFINITY_MASK:-?}}  regrid=${TF_REGRID:-gpu}  $(date)"

declare -A T
stage() {   # stage <name> <timeout s> <script> -- run, time, stop on failure
    local name=$1 tmo=$2 script=$3 t0=$SECONDS
    timeout "$tmo" "$PY" "$TF/$script" > "$name.log" 2>&1
    local rc=$?
    T[$name]=$((SECONDS - t0))
    if [ $rc -ne 0 ]; then
        echo "!!! $name failed (rc=$rc) after ${T[$name]}s"; tail -5 "$name.log"
        return 1
    fi
    echo "    $name: ${T[$name]}s"
}
write_times() {
    { echo "{"; local sep=""
      for k in "${!T[@]}"; do echo "$sep \"$k\": ${T[$k]}"; sep=","; done
      echo "}"; } > stage_times.json
}
trap write_times EXIT

stage blocks 3600 tf_aster_blocks.py || exit 1
echo "    blocks: $("$PY" -c "import json;print(len(json.load(open('aster_blocks.json'))))")"
stage sources 7200 tf_sources.py || exit 1
export TF_NPZ=$(ls regrid_inputs_*.npz | head -1)
stage congruence 7200 tf_congruence.py || exit 1
stage contrast 1800 tf_contrast.py || exit 1
echo "    congruence: $("$PY" -c "import json;print(len(json.load(open('congruence_b31_WN.json'))))") blocks  OK"
