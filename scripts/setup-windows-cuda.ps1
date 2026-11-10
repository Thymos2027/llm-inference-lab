$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$local = Join-Path $root '.local'
$envPrefix = Join-Path $local 'cuda-13.3'
$statusPath = Join-Path $local 'setup-status.json'
New-Item -ItemType Directory -Path $local -Force | Out-Null

function Set-Status([string]$phase, [string]$detail) {
    [ordered]@{
        phase = $phase
        detail = $detail
        updated = (Get-Date).ToString('o')
        environment = $envPrefix
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
    Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $phase - $detail"
}

function Get-Sha256([string]$path) {
    $stream = [System.IO.File]::OpenRead($path)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return -join ($sha.ComputeHash($stream) | ForEach-Object { $_.ToString('x2') })
    } finally {
        $sha.Dispose()
        $stream.Dispose()
    }
}

try {
    $cache = Join-Path $local 'conda-pkgs'
    New-Item -ItemType Directory -Path $cache -Force | Out-Null
    $packageName = 'cuda-nvvm-tools-13.3.73-haac1185_0.conda'
    $packagePath = Join-Path $cache $packageName
    $downloadPath = "$packagePath.download"
    $chunkPath = "$packagePath.chunk"
    $headerPath = "$packagePath.headers"
    $expectedSize = 45363795
    $expectedSha256 = '0f0ad672444906d4e80192c0b8c283bce927be9291730ed80df16c8b395d6c9e'
    $packageUrl = "https://conda.anaconda.org/nvidia/win-64/$packageName"
    $validPackage = (Test-Path -LiteralPath $packagePath) -and
        ((Get-Item -LiteralPath $packagePath).Length -eq $expectedSize) -and
        ((Get-Sha256 $packagePath) -eq $expectedSha256)
    if (-not $validPackage) {
        $chunkSize = 4MB
        $downloaded = 0
        if (Test-Path -LiteralPath $downloadPath) {
            $downloaded = (Get-Item -LiteralPath $downloadPath).Length
            if ($downloaded -gt $expectedSize -or
                ($downloaded -ne $expectedSize -and $downloaded % $chunkSize -ne 0)) {
                throw "Unexpected partial download length: $downloaded"
            }
        }
        while ($downloaded -lt $expectedSize) {
            $end = [Math]::Min($downloaded + $chunkSize - 1, $expectedSize - 1)
            $needed = $end - $downloaded + 1
            Set-Status 'downloading' "$packageName - $downloaded / $expectedSize bytes"
            $chunkValid = $false
            for ($attempt = 1; $attempt -le 8; $attempt++) {
                & curl.exe --location --range "$downloaded-$end" --fail --silent --show-error `
                    --connect-timeout 30 --max-time 360 --dump-header $headerPath `
                    --output $chunkPath $packageUrl
                $headers = if (Test-Path -LiteralPath $headerPath) {
                    Get-Content -LiteralPath $headerPath -Raw
                } else { '' }
                $rangePattern = "Content-Range:\s*bytes\s+$downloaded-$end/$expectedSize"
                $chunkValid = ($LASTEXITCODE -eq 0) -and
                    (Test-Path -LiteralPath $chunkPath) -and
                    ((Get-Item -LiteralPath $chunkPath).Length -eq $needed) -and
                    ($headers -match $rangePattern)
                if ($chunkValid) { break }
                Write-Output "Chunk $downloaded-$end attempt $attempt failed; retrying"
                Start-Sleep -Seconds 5
            }
            if (-not $chunkValid) { throw "Could not download chunk $downloaded-$end" }
            $destination = [System.IO.File]::Open($downloadPath, 'OpenOrCreate', 'Write', 'None')
            $source = [System.IO.File]::OpenRead($chunkPath)
            try {
                [void]$destination.Seek(0, [System.IO.SeekOrigin]::End)
                $source.CopyTo($destination)
            } finally {
                $source.Dispose()
                $destination.Dispose()
            }
            $downloaded = (Get-Item -LiteralPath $downloadPath).Length
        }
        $actualSha256 = Get-Sha256 $downloadPath
        if ($actualSha256 -ne $expectedSha256) {
            throw "SHA-256 mismatch for ${downloadPath}: $actualSha256"
        }
        Move-Item -LiteralPath $downloadPath -Destination $packagePath -Force
    }

    Set-Status 'installing' 'Installing CUDA 13.3 compiler and runtime from the verified package cache'
    $env:CONDA_PKGS_DIRS = $cache
    $env:CONDA_REMOTE_MAX_RETRIES = '8'
    $env:CONDA_REMOTE_BACKOFF_FACTOR = '2'
    $conda = (Get-Command conda.exe -ErrorAction Stop).Source
    $installedPackage = Get-ChildItem -Path (Join-Path $envPrefix 'conda-meta') -Filter 'cuda-nvvm-tools-*.json' -ErrorAction SilentlyContinue
    if (-not $installedPackage) {
        & $conda create --offline -y -p $envPrefix $packagePath
        if ($LASTEXITCODE -ne 0) { throw "Conda local package install exited with code $LASTEXITCODE" }
    }
    & $conda install -y -p $envPrefix -c nvidia 'cuda-nvcc=13.3' 'cuda-cudart=13.3' 'cuda-cccl=13.3'
    if ($LASTEXITCODE -ne 0) { throw "Conda dependency install exited with code $LASTEXITCODE" }

    Set-Status 'verifying' 'Checking nvcc, compiling and running experiment 01'
    & (Join-Path $PSScriptRoot 'build-windows.ps1')
    if ($LASTEXITCODE -ne 0) { throw "Build exited with code $LASTEXITCODE" }
    Set-Status 'complete' 'CUDA environment installed; experiment 01 compiled and passed'
} catch {
    Set-Status 'failed' $_.Exception.Message
    Write-Output $_.Exception.ToString()
    exit 1
}
