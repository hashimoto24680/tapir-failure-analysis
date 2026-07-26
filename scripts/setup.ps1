[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TapnetPath = Join-Path $ProjectRoot "external\tapnet"
$CheckpointPath = Join-Path $ProjectRoot "checkpoints\tapir_checkpoint_panning.pt"
$CheckpointPartial = "$CheckpointPath.partial"
$TapnetCommit = "989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc"
$CheckpointUrl = "https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt"
$CheckpointSha256 = "628611c656b3bd65d4a70fbf5526b62afe82d1b085ce6044685287fb78509daa"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $Arguments"
    }
}

function Find-Git {
    $Command = Get-Command git -ErrorAction SilentlyContinue
    if ($null -ne $Command) {
        return $Command.Source
    }
    foreach ($Candidate in @(
        "C:\Program Files\Git\cmd\git.exe",
        "$env:LOCALAPPDATA\Programs\Git\cmd\git.exe"
    )) {
        if (Test-Path -LiteralPath $Candidate) {
            return $Candidate
        }
    }
    throw "Git was not found. Install Git for Windows, then retry."
}

Push-Location $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Invoke-Checked "py" "-3.11" "-m" "venv" (Join-Path $ProjectRoot ".venv")
    }

    Invoke-Checked $VenvPython "-m" "pip" "install" "pip==26.1.2"
    Invoke-Checked $VenvPython "-m" "pip" "install" `
        "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" `
        "--index-url" "https://download.pytorch.org/whl/cu130"
    Invoke-Checked $VenvPython (Join-Path $ProjectRoot "scripts\check_cuda.py") `
        "--output" (Join-Path $ProjectRoot "results\setup\cuda-smoke.json")
    Invoke-Checked $VenvPython "-m" "pip" "install" `
        "--requirement" (Join-Path $ProjectRoot "requirements-lock.txt")

    $Git = Find-Git
    if (-not (Test-Path -LiteralPath (Join-Path $TapnetPath ".git"))) {
        if (Test-Path -LiteralPath $TapnetPath) {
            throw "$TapnetPath exists but is not a Git checkout. Move it aside and retry."
        }
        New-Item -ItemType Directory -Force -Path (Split-Path $TapnetPath) | Out-Null
        Invoke-Checked $Git "clone" "--filter=blob:none" `
            "https://github.com/google-deepmind/tapnet.git" $TapnetPath
    }
    Invoke-Checked $Git "-C" $TapnetPath "fetch" "--depth=1" "origin" $TapnetCommit
    Invoke-Checked $Git "-C" $TapnetPath "checkout" "--detach" $TapnetCommit
    $ActualCommit = (& $Git -C $TapnetPath rev-parse HEAD).Trim()
    if ($ActualCommit -ne $TapnetCommit) {
        throw "TapNet commit mismatch: expected $TapnetCommit, got $ActualCommit"
    }
    $TapnetChanges = (& $Git -C $TapnetPath status --porcelain) -join "`n"
    if ($TapnetChanges.Trim()) {
        throw "TapNet checkout has local changes. Use a clean checkout before setup."
    }

    & $VenvPython -m pip show tapnet *> $null
    if ($LASTEXITCODE -eq 0) {
        Invoke-Checked $VenvPython "-m" "pip" "uninstall" "--yes" "tapnet"
    }
    $TapnetEggInfo = [System.IO.Path]::GetFullPath((Join-Path $TapnetPath "tapnet.egg-info"))
    $ResolvedTapnetPath = [System.IO.Path]::GetFullPath($TapnetPath)
    if (Test-Path -LiteralPath $TapnetEggInfo) {
        if (-not $TapnetEggInfo.StartsWith(
            $ResolvedTapnetPath + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Refusing to remove package metadata outside TapNet: $TapnetEggInfo"
        }
        Remove-Item -LiteralPath $TapnetEggInfo -Recurse
    }
    $SitePackages = (& $VenvPython -c "import site; print(site.getsitepackages()[0])").Trim()
    $ExpectedVenvRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot ".venv"))
    $ResolvedSitePackages = [System.IO.Path]::GetFullPath($SitePackages)
    if (-not $ResolvedSitePackages.StartsWith($ExpectedVenvRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to write outside the project venv: $ResolvedSitePackages"
    }
    Set-Content -LiteralPath (Join-Path $ResolvedSitePackages "tapnet-pinned-source.pth") `
        -Value $TapnetPath -Encoding ASCII
    Invoke-Checked $VenvPython "-m" "pip" "install" "--no-deps" `
        "--no-build-isolation" "--editable" $ProjectRoot

    New-Item -ItemType Directory -Force -Path (Split-Path $CheckpointPath) | Out-Null
    if (Test-Path -LiteralPath $CheckpointPath) {
        $ActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $CheckpointPath).Hash.ToLowerInvariant()
        if ($ActualHash -ne $CheckpointSha256) {
            throw "Existing checkpoint hash mismatch: $ActualHash"
        }
    } else {
        Invoke-Checked "curl.exe" "--fail" "--location" `
            "--output" $CheckpointPartial $CheckpointUrl
        $ActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $CheckpointPartial).Hash.ToLowerInvariant()
        if ($ActualHash -ne $CheckpointSha256) {
            throw "Downloaded checkpoint hash mismatch: $ActualHash"
        }
        Move-Item -LiteralPath $CheckpointPartial -Destination $CheckpointPath
    }

    Invoke-Checked $VenvPython "-c" `
        "from tapnet.torch import tapir_model; print(tapir_model.__file__)"
    Invoke-Checked $VenvPython "-m" "pip" "check"
    Write-Host "Setup complete. Run:"
    Write-Host ".\.venv\Scripts\python.exe scripts\run_tapir_smoke.py"
} finally {
    Pop-Location
}
