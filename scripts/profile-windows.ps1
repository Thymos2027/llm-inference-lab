param(
    [ValidateSet('VectorAdd')][string]$Experiment = 'VectorAdd'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

if ($Experiment -ne 'VectorAdd') {
    throw "Unsupported experiment: $Experiment"
}

& (Join-Path $PSScriptRoot 'build-windows.ps1') -Lesson Nsight -BuildOnly
if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }

$executable = Join-Path $root '01-cuda-matmul\05-nsight-profiling\vector_add.exe'
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

$reportDir = Join-Path $root '.local\profiles\05-vector-add'
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
$nsysReport = Join-Path $reportDir 'vector-add-systems'
$ncuReport = Join-Path $reportDir 'vector-add-compute'

Write-Host 'Collecting the CUDA timeline with Nsight Systems...'
& $nsys profile --trace=cuda --sample=none --force-overwrite=true --output=$nsysReport $executable --profile
if ($LASTEXITCODE -ne 0) { throw "Nsight Systems failed with exit code $LASTEXITCODE" }

Write-Host 'Collecting kernel metrics with Nsight Compute...'
& $ncu --section LaunchStats --section Occupancy --section SpeedOfLight --section MemoryWorkloadAnalysis --force-overwrite --export $ncuReport $executable --profile
if ($LASTEXITCODE -ne 0) { throw "Nsight Compute failed with exit code $LASTEXITCODE" }

Write-Host "Nsight Systems report: $nsysReport.nsys-rep"
Write-Host "Nsight Compute report: $ncuReport.ncu-repz"
