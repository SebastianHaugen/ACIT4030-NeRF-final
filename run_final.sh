#!/usr/bin/env bash
# Final ablation on the corrected data (lego colours multiplied by alpha, poster undistorted),
# written to outputs/src_fixed. The A2 baseline was trained on the uncorrected data, so the book
# model is retrained here ('uniform') as the reference row.
# Runs the most important rows first (book model and hash grid on both datasets), then the rest.
#
# Usage, from anywhere:
#     PYTHON=/path/to/python ./run_final.sh            # train + evaluate everything not finished yet
#     PYTHON=/path/to/python ./run_final.sh --smoke    # 1000-iteration smoke test of every row
# Long runs: start with nohup / setsid, or inside tmux. Follow: tail -f logs/final_all.log

set -u
MODE="${1:-full}"
PYTHON="${PYTHON:-python}"
cd "$(dirname "$0")"
mkdir -p logs
unset $(compgen -v NERF_) 2>/dev/null
export NERF_FIX_DATA=true

# Book model = the A2 baseline (config.yaml defaults), retrained on the corrected data
BOOK=""
# + spacing-aware density (the big poster gain in the first ablation)
DELTA="NERF_DELTA=true"
# + NeRF encoding (L = 10 / 4) and 8-layer MLP, stratified sampling
MLP="NERF_DELTA=true NERF_STRATIFIED=true NERF_N_HARM=10 NERF_N_HARM_DIR=4 NERF_OMEGA0=3.14159265 NERF_INCLUDE_INPUT=true NERF_ARCH=nerf NERF_LR=0.0005"
# + proposal sampling (128 -> 64) and distortion loss
PROP="$MLP NERF_SAMPLER=proposal NERF_N_COARSE=128 NERF_N_FINE=64 NERF_DIST_WEIGHT=0.01"
# + smooth learning rate and 4096 rays per step
BUDGET="NERF_LR=0.001 NERF_LR_SCHEDULE=exp NERF_LR_FINAL=0.00001 NERF_LR_WARMUP=500 NERF_N_RAYS=1024"
# (poster) + lin-disp spacing, scene contraction, appearance embeddings
UNBOUNDED="NERF_SPACING=lindisp NERF_CONTRACT=true NERF_MAX_DEPTH=50 NERF_APPEARANCE_DIM=16"
# Hash grid (Instant-NGP encoding + tiny MLPs) with a hash-grid proposal network, as nerfacto:
# lr 1e-2 -> 1e-4, Adam eps 1e-15, distortion weight 0.002, 4096 rays per step
HASH="NERF_ARCH=hashgrid NERF_DELTA=true NERF_STRATIFIED=true NERF_SAMPLER=proposal NERF_N_COARSE=128 NERF_N_FINE=64 NERF_N_HARM_DIR=4 NERF_OMEGA0=3.14159265 NERF_INCLUDE_INPUT=true NERF_DIST_WEIGHT=0.002 NERF_LR=0.01 NERF_LR_SCHEDULE=exp NERF_LR_FINAL=0.0001 NERF_LR_WARMUP=100 NERF_ADAM_EPS=1e-15 NERF_N_RAYS=1024"

run_src() {
    local dataset="$1" name="$2" settings="$3"
    local log="logs/final_${dataset}_${name}.log"
    if [[ "$MODE" == "--smoke" ]]; then
        log="logs/final_${dataset}_${name}_smoke.log"
        # shellcheck disable=SC2086  # the settings are meant to be split into separate variables
        env NERF_DATASET="$dataset" NERF_EXPERIMENT="$name" $settings "$PYTHON" -m src.smoke_test > "$log" 2>&1
        # Checked in the log, not the exit code: Python sometimes segfaults while shutting down CUDA
        # after everything has finished, which gives a non-zero exit code for a run that passed
        if grep -q "^PASS" "$log"; then
            echo "$(date +%H:%M)  PASS  $dataset $name  ($(grep -o 'Loss: .*' "$log"), $(grep -o '[0-9.]* s/iteration' "$log"))"
        else
            echo "$(date +%H:%M)  FAIL  $dataset $name  (see $log)"
        fi
        return
    fi
    if [[ -f "outputs/src_fixed/$dataset/$name/metrics.csv" ]]; then
        echo "$(date +%H:%M)  skip  $dataset $name  (already finished)"
        return
    fi
    echo "$(date +%H:%M)  start $dataset $name  (log: $log)"
    rm -f "outputs/src_fixed/$dataset/$name/loss_history.csv"
    # shellcheck disable=SC2086
    env NERF_DATASET="$dataset" NERF_EXPERIMENT="$name" $settings "$PYTHON" -m src.train > "$log" 2>&1
    # loss_history.csv is written at the very end of training (see the smoke-test note above)
    if [[ ! -f "outputs/src_fixed/$dataset/$name/loss_history.csv" ]]; then
        echo "$(date +%H:%M)  FAILED in training: $dataset $name  (see $log)"
        return
    fi
    # shellcheck disable=SC2086
    env NERF_DATASET="$dataset" NERF_EXPERIMENT="$name" $settings "$PYTHON" -m src.evaluate >> "$log" 2>&1
    echo "$(date +%H:%M)  done  $dataset $name  ($(grep -E '^PSNR' "$log"))"
}

start=$(date +%s)

# Most important first: the reference and the hash grid on both datasets
run_src lego   "uniform"  "$BOOK"
run_src lego   "hashgrid" "$HASH"
run_src poster "uniform"  "$BOOK"
run_src poster "hashgrid" "$HASH $UNBOUNDED"

# The steps in between, so the table shows what each change adds
run_src lego   "uniform_delta"                "$DELTA"
run_src lego   "enc_mlp"                      "$MLP"
run_src lego   "enc_mlp_dist_prop"            "$PROP"
run_src lego   "enc_mlp_dist_prop_exp_4096"   "$PROP $BUDGET"
run_src poster "uniform_delta"                "$DELTA"
run_src poster "enc_mlp"                      "$MLP"
run_src poster "enc_mlp_dist_prop"            "$PROP"
run_src poster "enc_mlp_dist_prop_contract_app"           "$PROP $UNBOUNDED"
run_src poster "enc_mlp_dist_prop_contract_app_exp_4096"  "$PROP $UNBOUNDED $BUDGET"

# Lego is the dynamic D-NeRF scene (the bucket moves), so the best lego models are retrained with
# the frame time as an extra input (T-NeRF)
run_src lego   "hashgrid_time"                   "$HASH NERF_TIME=true"
run_src lego   "enc_mlp_dist_prop_exp_4096_time" "$PROP $BUDGET NERF_TIME=true"

if [[ "$MODE" != "--smoke" ]]; then
    NERF_FIX_DATA=true "$PYTHON" -m src.report_figures > logs/final_report.log 2>&1
    cat logs/final_report.log
fi
echo "All runs finished in $(( ($(date +%s) - start) / 60 )) min"
