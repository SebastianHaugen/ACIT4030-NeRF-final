#!/usr/bin/env bash
# Linux version of run_improvements.ps1: the second ablation (model + mip-NeRF 360 additions),
# one change per row on top of the row before, then the report figures.
#
# Usage, from anywhere:
#     ./run_improvements.sh lego             # full runs: train + evaluate every row
#     ./run_improvements.sh poster
#     ./run_improvements.sh poster --smoke   # 1000-iteration smoke test of every row first
# Pick the Python environment with PYTHON=/path/to/python (default: python).
# Long runs: start inside tmux, or with nohup ... & so closing the terminal does not stop them.
# Follow progress:  tail -f logs/<dataset>_<name>.log

set -u
DATASET="${1:-lego}"
MODE="${2:-full}"
PYTHON="${PYTHON:-python}"
if [[ "$DATASET" != "lego" && "$DATASET" != "poster" ]]; then
    echo "Usage: $0 lego|poster [--smoke]" >&2
    exit 1
fi

cd "$(dirname "$0")"
mkdir -p logs
# Removes NERF_* variables left in the shell, so only the settings below are used
unset $(compgen -v NERF_) 2>/dev/null
export NERF_DATASET="$DATASET"
# This ablation was run before the data fixes, so it keeps using the uncorrected data (outputs/src).
# The ablation on the corrected data is run_final.sh
export NERF_FIX_DATA=false

# Starting point: the best cheap row of the sampling ablation (uniform, stratified, delta, 128 points)
SAMPLING="NERF_SAMPLER=uniform NERF_STRATIFIED=true NERF_DELTA=true NERF_N_PTS=128"
# 1. NeRF positional encoding: L = 10 for positions, 4 for directions, lowest frequency pi, raw xyz included
ENC="$SAMPLING NERF_N_HARM=10 NERF_N_HARM_DIR=4 NERF_OMEGA0=3.14159265 NERF_INCLUDE_INPUT=true"
# 2. + NeRF MLP (8 layers, skip connection). NeRF trains it with lr 5e-4
MLP="$ENC NERF_ARCH=nerf NERF_LR=0.0005"
# 3. + distortion loss (mip-NeRF 360, weight 0.01 as in the paper)
DIST="$MLP NERF_DIST_WEIGHT=0.01"
# 4. + proposal sampling: 128 proposal samples, 64 NeRF samples (later values replace earlier ones)
PROP="$DIST NERF_SAMPLER=proposal NERF_N_COARSE=128 NERF_N_FINE=64"
# 5. (poster) + lin-disp spacing and scene contraction, so the rays can reach much further into the room
CONTRACT="$PROP NERF_SPACING=lindisp NERF_CONTRACT=true NERF_MAX_DEPTH=50"
# 6. (poster) + appearance embeddings, against exposure / white balance changes between the photos
APP="$CONTRACT NERF_APPEARANCE_DIM=16"
# Training budget, added on top of the best model of each dataset (row 4 for lego, row 6 for poster):
# + smooth learning rate: 500 warm-up iterations, then log-linear decay 1e-3 -> 1e-5 (mip-NeRF 360 style)
EXP="NERF_LR=0.001 NERF_LR_SCHEDULE=exp NERF_LR_FINAL=0.00001 NERF_LR_WARMUP=500"
# + 4096 rays per step (4 images x 1024 rays) as in NeRF, instead of 1600
RAYS="$EXP NERF_N_RAYS=1024"

run_src() {
    local name="$1" settings="$2"
    local log="logs/${DATASET}_${name}.log"
    if [[ "$MODE" == "--smoke" ]]; then
        log="logs/${DATASET}_${name}_smoke.log"
        echo "$(date +%H:%M)  smoke $name  (log: $log)"
        # shellcheck disable=SC2086  # the settings are meant to be split into separate variables
        env NERF_EXPERIMENT="$name" $settings "$PYTHON" -m src.smoke_test > "$log" 2>&1
        if grep -q "^PASS" "$log"; then
            echo "$(date +%H:%M)  PASS  $name  ($(grep -o 'Loss: .*' "$log"))"
        else
            echo "$(date +%H:%M)  FAIL  $name  (see $log)"
        fi
        return
    fi
    if [[ -f "outputs/src/$DATASET/$name/metrics.csv" ]]; then
        echo "$(date +%H:%M)  skip  $name  (already finished, delete outputs/src/$DATASET/$name to rerun)"
        return
    fi
    echo "$(date +%H:%M)  start $name  (log: $log)"
    rm -f "outputs/src/$DATASET/$name/loss_history.csv"
    # shellcheck disable=SC2086
    env NERF_EXPERIMENT="$name" $settings "$PYTHON" -m src.train > "$log" 2>&1
    # Checked by the output file, not the exit code: Python sometimes segfaults while shutting
    # down CUDA after a finished run. loss_history.csv is written at the very end of training
    if [[ ! -f "outputs/src/$DATASET/$name/loss_history.csv" ]]; then
        echo "$(date +%H:%M)  FAILED in training: $name  (see $log)"
        return
    fi
    # shellcheck disable=SC2086
    env NERF_EXPERIMENT="$name" $settings "$PYTHON" -m src.evaluate >> "$log" 2>&1
    echo "$(date +%H:%M)  done  $name  ($(grep -E '^PSNR' "$log"))"
}

start=$(date +%s)

run_src "enc_10_4"          "$ENC"
run_src "enc_mlp"           "$MLP"
run_src "enc_mlp_dist"      "$DIST"
run_src "enc_mlp_dist_prop" "$PROP"
if [[ "$DATASET" == "lego" ]]; then
    run_src "enc_mlp_dist_prop_exp"      "$PROP $EXP"
    run_src "enc_mlp_dist_prop_exp_4096" "$PROP $RAYS"
fi
if [[ "$DATASET" == "poster" ]]; then
    run_src "enc_mlp_dist_prop_contract"     "$CONTRACT"
    run_src "enc_mlp_dist_prop_contract_app" "$APP"
    run_src "enc_mlp_dist_prop_contract_app_exp"      "$APP $EXP"
    run_src "enc_mlp_dist_prop_contract_app_exp_4096" "$APP $RAYS"
fi

if [[ "$MODE" != "--smoke" ]]; then
    # Table + loss plots for everything that finished
    "$PYTHON" -m src.report_figures > "logs/${DATASET}_report_improvements.log" 2>&1
    cat "logs/${DATASET}_report_improvements.log"
fi

echo "All $DATASET runs finished in $(( ($(date +%s) - start) / 60 )) min"
