# Windows version of run_final.sh: the final ablation on the corrected data (outputs\src_fixed).
# Runs the most important rows first (book model and hash grid on both datasets), then the rest.
#     powershell -ExecutionPolicy Bypass -File run_final.ps1
# Follow progress:  Get-Content logs\final_<dataset>_<name>.log -Tail 5 -Wait

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null

function Clear-Overrides {
    Get-ChildItem Env: | Where-Object { $_.Name -like "NERF_*" } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
}

function Merge($a, $b) {
    $c = $a.Clone()
    foreach ($key in $b.Keys) { $c[$key] = $b[$key] }
    return $c
}

function Run-Src($dataset, $name, $overrides) {
    if (Test-Path "outputs\src_fixed\$dataset\$name\metrics.csv") {
        Write-Host "$(Get-Date -Format HH:mm)  skip  $dataset $name  (already finished)"
        return
    }
    Clear-Overrides
    $env:NERF_FIX_DATA = "true"
    $env:NERF_DATASET = $dataset
    $env:NERF_EXPERIMENT = $name
    foreach ($key in $overrides.Keys) { Set-Item "Env:$key" $overrides[$key] }
    $log = "logs\final_${dataset}_$name.log"
    Write-Host "$(Get-Date -Format HH:mm)  start $dataset $name  (log: $log)"
    python -m src.train *> $log
    python -m src.evaluate *>> $log
    Write-Host "$(Get-Date -Format HH:mm)  done  $dataset $name"
}

$start = Get-Date

$book = @{}
$delta = @{ NERF_DELTA = "true" }
$mlp = @{ NERF_DELTA = "true"; NERF_STRATIFIED = "true"; NERF_N_HARM = "10"; NERF_N_HARM_DIR = "4";
          NERF_OMEGA0 = "3.14159265"; NERF_INCLUDE_INPUT = "true"; NERF_ARCH = "nerf"; NERF_LR = "0.0005" }
$prop = Merge $mlp @{ NERF_SAMPLER = "proposal"; NERF_N_COARSE = "128"; NERF_N_FINE = "64"; NERF_DIST_WEIGHT = "0.01" }
$budget = @{ NERF_LR = "0.001"; NERF_LR_SCHEDULE = "exp"; NERF_LR_FINAL = "0.00001"; NERF_LR_WARMUP = "500"; NERF_N_RAYS = "1024" }
$unbounded = @{ NERF_SPACING = "lindisp"; NERF_CONTRACT = "true"; NERF_MAX_DEPTH = "50"; NERF_APPEARANCE_DIM = "16" }
$hash = @{ NERF_ARCH = "hashgrid"; NERF_DELTA = "true"; NERF_STRATIFIED = "true"; NERF_SAMPLER = "proposal";
           NERF_N_COARSE = "128"; NERF_N_FINE = "64"; NERF_N_HARM_DIR = "4"; NERF_OMEGA0 = "3.14159265";
           NERF_INCLUDE_INPUT = "true"; NERF_DIST_WEIGHT = "0.002"; NERF_LR = "0.01"; NERF_LR_SCHEDULE = "exp";
           NERF_LR_FINAL = "0.0001"; NERF_LR_WARMUP = "100"; NERF_ADAM_EPS = "1e-15"; NERF_N_RAYS = "1024" }

# Most important first: the reference and the hash grid on both datasets
Run-Src "lego"   "uniform"  $book
Run-Src "lego"   "hashgrid" $hash
Run-Src "poster" "uniform"  $book
Run-Src "poster" "hashgrid" (Merge $hash $unbounded)

# The steps in between
Run-Src "lego"   "uniform_delta"               $delta
Run-Src "lego"   "enc_mlp"                     $mlp
Run-Src "lego"   "enc_mlp_dist_prop"           $prop
Run-Src "lego"   "enc_mlp_dist_prop_exp_4096"  (Merge $prop $budget)
Run-Src "poster" "uniform_delta"               $delta
Run-Src "poster" "enc_mlp"                     $mlp
Run-Src "poster" "enc_mlp_dist_prop"           $prop
Run-Src "poster" "enc_mlp_dist_prop_contract_app"           (Merge $prop $unbounded)
Run-Src "poster" "enc_mlp_dist_prop_contract_app_exp_4096"  (Merge (Merge $prop $unbounded) $budget)

# Lego is the dynamic D-NeRF scene (the bucket moves), so the best lego models are retrained with
# the frame time as an extra input (T-NeRF)
Run-Src "lego"   "hashgrid_time"                    (Merge $hash @{ NERF_TIME = "true" })
Run-Src "lego"   "enc_mlp_dist_prop_exp_4096_time"  (Merge (Merge $prop $budget) @{ NERF_TIME = "true" })

Clear-Overrides
$env:NERF_FIX_DATA = "true"
python -m src.report_figures *> logs\final_report.log
Get-Content logs\final_report.log
Write-Host "All runs finished in $([math]::Round(((Get-Date) - $start).TotalHours, 1)) h"
