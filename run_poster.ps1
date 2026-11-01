# Runs the full poster ablation back to back: baseline + 5 src experiments, then the report figures.
# Start from the repo root ON THE STATIONARY PC (not over SSH, which stops programs when it disconnects):
#     powershell -ExecutionPolicy Bypass -File run_poster.ps1
# Follow progress from the laptop over SSH:
#     Get-Content logs\poster_<name>.log -Tail 5 -Wait

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$env:NERF_DATASET = "poster"

function Clear-Overrides {
    foreach ($name in "NERF_EXPERIMENT", "NERF_SAMPLER", "NERF_STRATIFIED", "NERF_DELTA",
                      "NERF_N_COARSE", "NERF_N_FINE", "NERF_N_PTS") {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

function Run-Src($name, $sampler, $stratified, $delta, $nPts) {
    Clear-Overrides
    $env:NERF_EXPERIMENT = $name
    $env:NERF_SAMPLER = $sampler
    $env:NERF_STRATIFIED = $stratified
    $env:NERF_DELTA = $delta
    $env:NERF_N_PTS = $nPts
    $log = "logs\poster_$name.log"
    Write-Host "$(Get-Date -Format HH:mm)  start $name  (log: $log)"
    python -m src.train *> $log
    python -m src.evaluate *>> $log
    Write-Host "$(Get-Date -Format HH:mm)  done  $name"
}

$start = Get-Date

# 1. Frozen baseline (reads baseline/config.yaml only)
Clear-Overrides
Write-Host "$(Get-Date -Format HH:mm)  start baseline  (log: logs\poster_baseline.log)"
python -m baseline.train *> logs\poster_baseline.log
python -m baseline.evaluate *>> logs\poster_baseline.log
Write-Host "$(Get-Date -Format HH:mm)  done  baseline"

# 2-6. src ablation rows:   name                       sampler         strat    delta    points per ray (uniform)
Run-Src                     "uniform"                  "uniform"       "false"  "false"  128
Run-Src                     "uniform_delta"            "uniform"       "false"  "true"   128
Run-Src                     "uniform_delta_strat"      "uniform"       "true"   "true"   128
Run-Src                     "uniform_delta_strat_192"  "uniform"       "true"   "true"   192
Run-Src                     "hier_64_128"              "hierarchical"  "true"   "true"   128

# Table + loss plots for everything that finished
Clear-Overrides
python -m src.report_figures *> logs\poster_report.log
Get-Content logs\poster_report.log

Write-Host "All poster runs finished in $([math]::Round(((Get-Date) - $start).TotalHours, 1)) h"