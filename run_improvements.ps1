# Runs the second ablation (model + mip-NeRF 360 additions) back to back, then the report figures.
# Each row adds one change on top of the row before, so the table shows what each change adds.
# Start from the repo root ON THE STATIONARY PC (not over SSH, which stops programs when it disconnects):
#     powershell -ExecutionPolicy Bypass -File run_improvements.ps1 -Dataset lego
#     powershell -ExecutionPolicy Bypass -File run_improvements.ps1 -Dataset poster
# Follow progress from the laptop over SSH:
#     Get-Content logs\<dataset>_<name>.log -Tail 5 -Wait
# Tip: run the smoke test of a row first by setting the same variables and running python -m src.smoke_test

param(
    [ValidateSet("lego", "poster")]
    [string]$Dataset = "lego"
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$env:NERF_DATASET = $Dataset
# This ablation was run before the data fixes, so it keeps using the uncorrected data (outputs\src).
# The ablation on the corrected data is run_final.ps1
$env:NERF_FIX_DATA = "false"

$OverrideNames = @(
    "NERF_EXPERIMENT", "NERF_SAMPLER", "NERF_STRATIFIED", "NERF_DELTA", "NERF_N_COARSE", "NERF_N_FINE",
    "NERF_N_PTS", "NERF_SPACING", "NERF_CONTRACT", "NERF_MAX_DEPTH", "NERF_ARCH", "NERF_N_HARM",
    "NERF_N_HARM_DIR", "NERF_OMEGA0", "NERF_INCLUDE_INPUT", "NERF_APPEARANCE_DIM", "NERF_LR",
    "NERF_DIST_WEIGHT", "NERF_LR_SCHEDULE", "NERF_LR_FINAL", "NERF_LR_WARMUP", "NERF_N_RAYS", "NERF_BATCH_SIZE"
)

function Clear-Overrides {
    foreach ($name in $OverrideNames) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

# Copy of $a with the values of $b added or replaced
function Merge($a, $b) {
    $c = $a.Clone()
    foreach ($key in $b.Keys) { $c[$key] = $b[$key] }
    return $c
}

function Run-Src($name, $overrides) {
    if (Test-Path "outputs\src\$Dataset\$name\metrics.csv") {
        Write-Host "$(Get-Date -Format HH:mm)  skip  $name  (already finished)"
        return
    }
    Clear-Overrides
    $env:NERF_EXPERIMENT = $name
    foreach ($key in $overrides.Keys) { Set-Item "Env:$key" $overrides[$key] }
    $log = "logs\${Dataset}_$name.log"
    Write-Host "$(Get-Date -Format HH:mm)  start $name  (log: $log)"
    python -m src.train *> $log
    python -m src.evaluate *>> $log
    Write-Host "$(Get-Date -Format HH:mm)  done  $name"
}

$start = Get-Date

# Starting point: the best cheap row of the sampling ablation (uniform, stratified, delta, 128 points)
$sampling = @{ NERF_SAMPLER = "uniform"; NERF_STRATIFIED = "true"; NERF_DELTA = "true"; NERF_N_PTS = "128" }

# 1. NeRF positional encoding: L = 10 for positions, 4 for directions, lowest frequency pi, raw xyz included
$enc = Merge $sampling @{ NERF_N_HARM = "10"; NERF_N_HARM_DIR = "4"; NERF_OMEGA0 = "3.14159265"; NERF_INCLUDE_INPUT = "true" }
# 2. + NeRF MLP (8 layers, skip connection). NeRF trains it with lr 5e-4
$mlp = Merge $enc @{ NERF_ARCH = "nerf"; NERF_LR = "0.0005" }
# 3. + distortion loss (mip-NeRF 360, weight 0.01 as in the paper)
$dist = Merge $mlp @{ NERF_DIST_WEIGHT = "0.01" }
# 4. + proposal sampling: 128 proposal samples, 64 NeRF samples (half the NeRF evaluations of row 3)
$prop = Merge $dist @{ NERF_SAMPLER = "proposal"; NERF_N_COARSE = "128"; NERF_N_FINE = "64" }

Run-Src "enc_10_4"            $enc
Run-Src "enc_mlp"             $mlp
Run-Src "enc_mlp_dist"        $dist
Run-Src "enc_mlp_dist_prop"   $prop

# Training budget on top of the best model: smooth learning rate (500 warm-up iterations, then
# log-linear decay 1e-3 -> 1e-5), then 4096 rays per step (4 images x 1024 rays) as in NeRF
$exp = @{ NERF_LR = "0.001"; NERF_LR_SCHEDULE = "exp"; NERF_LR_FINAL = "0.00001"; NERF_LR_WARMUP = "500" }
$rays = Merge $exp @{ NERF_N_RAYS = "1024" }

if ($Dataset -eq "lego") {
    Run-Src "enc_mlp_dist_prop_exp"       (Merge $prop $exp)
    Run-Src "enc_mlp_dist_prop_exp_4096"  (Merge $prop $rays)
}

if ($Dataset -eq "poster") {
    # 5. + lin-disp spacing and scene contraction, so the rays can reach much further into the room
    $contract = Merge $prop @{ NERF_SPACING = "lindisp"; NERF_CONTRACT = "true"; NERF_MAX_DEPTH = "50" }
    # 6. + appearance embeddings, against exposure / white balance changes between the photos
    $app = Merge $contract @{ NERF_APPEARANCE_DIM = "16" }

    Run-Src "enc_mlp_dist_prop_contract"      $contract
    Run-Src "enc_mlp_dist_prop_contract_app"  $app
    Run-Src "enc_mlp_dist_prop_contract_app_exp"       (Merge $app $exp)
    Run-Src "enc_mlp_dist_prop_contract_app_exp_4096"  (Merge $app $rays)
}

# Table + loss plots for everything that finished
Clear-Overrides
python -m src.report_figures *> "logs\${Dataset}_report_improvements.log"
Get-Content "logs\${Dataset}_report_improvements.log"

Write-Host "All $Dataset runs finished in $([math]::Round(((Get-Date) - $start).TotalHours, 1)) h"
