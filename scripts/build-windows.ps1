param(
    [ValidateSet('Naive', 'Coalesced', 'Tiled')][string]$Lesson = 'Naive',
    [switch]$Observe,
    [switch]$Benchmark
)

if (($Observe -or $Benchmark) -and $Lesson -ne 'Naive') {
    throw '-Observe and -Benchmark are only available for the Naive lesson'
}
if ($Observe -and $Benchmark) {
    throw 'Choose either -Observe or -Benchmark'
}

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$envPrefix = Join-Path $root '.local\cuda-13.3'
$nvcc = Get-ChildItem -LiteralPath $envPrefix -Filter nvcc.exe -Recurse -File -ErrorAction Stop |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $nvcc) { throw "nvcc.exe not found in $envPrefix" }

$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
if (-not $vsPath) { throw 'Visual Studio C++ Build Tools not found' }
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path -LiteralPath $vcvars)) { throw "Missing $vcvars" }

$lessonPath = switch ($Lesson) {
    'Naive' { '01-cuda-matmul\01-naive\sgemm_naive' }
    'Coalesced' { '01-cuda-matmul\02-coalesced-access\sgemm_access' }
    'Tiled' { '01-cuda-matmul\03-shared-memory-tiling\sgemm_tiled' }
}
$source = Join-Path $root "$lessonPath.cu"
$output = Join-Path $root "$lessonPath.exe"
$batch = Join-Path $root '.local\build-windows.cmd'
$run = if ($Observe) { "`"$output`" --observe" } elseif ($Benchmark) { "`"$output`" --benchmark" } else { "`"$output`"" }
@(
    '@echo off'
    "call `"$vcvars`" >nul"
    'if errorlevel 1 exit /b 1'
    "`"$nvcc`" -O2 -std=c++17 -arch=sm_120 -Xcompiler /utf-8 `"$source`" -o `"$output`""
    'if errorlevel 1 exit /b 1'
    $run
) | Set-Content -LiteralPath $batch -Encoding Ascii

& cmd.exe /d /c $batch
if ($LASTEXITCODE -ne 0) { throw "Build or validation failed with exit code $LASTEXITCODE" }
