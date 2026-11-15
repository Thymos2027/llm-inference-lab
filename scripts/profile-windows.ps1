param(
    [ValidateSet('VectorAdd', 'Transpose', 'Sgemm')][string]$Experiment = 'VectorAdd'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$lesson = switch ($Experiment) {
    'VectorAdd' { 'Nsight' }
    'Transpose' { 'Transpose' }
    'Sgemm' { 'SgemmCublas' }
}
& (Join-Path $PSScriptRoot 'build-windows.ps1') -Lesson $lesson -BuildOnly
if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }

$executableName = switch ($Experiment) {
    'VectorAdd' { 'vector_add.exe' }
    'Transpose' { 'transpose.exe' }
    'Sgemm' { 'sgemm_cublas.exe' }
}
$executable = Join-Path $root "01-cuda-matmul\05-nsight-profiling\$executableName"
if ($Experiment -eq 'Sgemm') {
    $env:PATH = "$(Join-Path $root '.local\cuda-13.3\Library\bin');$env:PATH"
}
$profilerRoot = Join-Path $root '.local\profilers'
$ncuPathFile = Join-Path $profilerRoot 'nsight-compute-path.txt'
$nsysPathFile = Join-Path $profilerRoot 'nsight-systems-path.txt'
foreach ($path in @($executable, $ncuPathFile, $nsysPathFile)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required file not found: $path" }
}

$ncu = (Get-Content -LiteralPath $ncuPathFile -Raw).Trim()
$nsys = (Get-Content -LiteralPath $nsysPathFile -Raw).Trim()
foreach ($path in @($ncu, $nsys)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Profiler not found: $path" }
}

$reportFolder = if ($Experiment -eq 'VectorAdd') { '05-vector-add' } else { "05-$($Experiment.ToLower())" }
$reportDir = Join-Path $root ".local\profiles\$reportFolder"
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$variants = switch ($Experiment) {
    'VectorAdd' { @('') }
    'Transpose' { @('copy', 'naive', 'shared') }
    'Sgemm' { @('naive', 'tiled', 'register') }
}
$memorySection = if ($Experiment -eq 'VectorAdd') { 'MemoryWorkloadAnalysis' } else { 'MemoryWorkloadAnalysis_Tables' }
foreach ($variant in $variants) {
    $label = if ($variant) { $variant } else { 'vector-add' }
    $profileArgs = @('--profile')
    if ($variant) { $profileArgs += $variant }
    $nsysReport = Join-Path $reportDir "$label-systems"
    $ncuReport = Join-Path $reportDir "$label-compute"

    Write-Host "Collecting $label CUDA timeline..."
    & $nsys profile --trace=cuda --sample=none --force-overwrite=true --output=$nsysReport $executable @profileArgs
    if ($LASTEXITCODE -ne 0) { throw "Nsight Systems failed with exit code $LASTEXITCODE" }

    Write-Host "Collecting $label kernel metrics..."
    & $ncu --section LaunchStats --section Occupancy --section SpeedOfLight --section $memorySection --force-overwrite --export $ncuReport $executable @profileArgs
    if ($LASTEXITCODE -ne 0) { throw "Nsight Compute failed with exit code $LASTEXITCODE" }

    Write-Host "Nsight Systems report: $nsysReport.nsys-rep"
    Write-Host "Nsight Compute report: $ncuReport.ncu-repz"
}
