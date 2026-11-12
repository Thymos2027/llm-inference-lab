param(
    [ValidateSet('Naive', 'Coalesced', 'Tiled', 'Register', 'Nsight', 'NsightPractice')][string]$Lesson = 'Naive',
    [switch]$Observe,
    [switch]$Benchmark,
    [switch]$BuildOnly,
    [ValidateSet(4, 8, 16, 32)][int]$ThreadTileM = 8,
    [ValidateRange(0, 16)][int]$KUnroll = 0
)

if (($Observe -or $Benchmark) -and $Lesson -ne 'Naive') {
    throw '-Observe and -Benchmark are only available for the Naive lesson'
}
if ($Observe -and $Benchmark) {
    throw 'Choose either -Observe or -Benchmark'
}
if ($BuildOnly -and ($Observe -or $Benchmark)) {
    throw '-BuildOnly cannot be combined with -Observe or -Benchmark'
}
if ($PSBoundParameters.ContainsKey('ThreadTileM') -and $Lesson -ne 'Register') {
    throw '-ThreadTileM is only available for the Register lesson'
}
if ($PSBoundParameters.ContainsKey('KUnroll') -and $Lesson -ne 'Register') {
    throw '-KUnroll is only available for the Register lesson'
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
    'Register' { '01-cuda-matmul\04-register-tiling\sgemm_register' }
    'Nsight' { '01-cuda-matmul\05-nsight-profiling\vector_add' }
    'NsightPractice' { '01-cuda-matmul\05-nsight-profiling\vector_add_practice' }
}
$source = Join-Path $root "$lessonPath.cu"
$output = Join-Path $root "$lessonPath.exe"
$batch = Join-Path $root '.local\build-windows.cmd'
$run = if ($BuildOnly) { '' } elseif ($Observe) { "`"$output`" --observe" } elseif ($Benchmark) { "`"$output`" --benchmark" } else { "`"$output`"" }
$tmDefine = if ($Lesson -eq 'Register') { "-DTHREAD_TILE_M=$ThreadTileM " } else { '' }
$unrollDefine = if ($Lesson -eq 'Register' -and $KUnroll -gt 0) { "-DREGISTER_K_UNROLL=$KUnroll " } else { '' }
$lineInfo = if ($Lesson -in @('Nsight', 'NsightPractice')) { '-lineinfo ' } else { '' }
@(
    '@echo off'
    "call `"$vcvars`" >nul"
    'if errorlevel 1 exit /b 1'
    "`"$nvcc`" -O2 -std=c++17 -arch=sm_120 -Xcompiler /utf-8 $lineInfo$tmDefine$unrollDefine`"$source`" -o `"$output`""
    'if errorlevel 1 exit /b 1'
    $run
) | Set-Content -LiteralPath $batch -Encoding Ascii

& cmd.exe /d /c $batch
if ($LASTEXITCODE -ne 0) { throw "Build or validation failed with exit code $LASTEXITCODE" }
