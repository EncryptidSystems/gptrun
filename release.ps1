# release.ps1 builds a ModelRun release zip that can be traced back to its source.
#
#   pwsh .\release.ps1 -Release 0.3.0
#
# Steps: refuse uncommitted changes (ModelRun and the hodgemix checkout) unless
# -AllowDirty; go vet and the unit tests; the GPU tests when MODELRUN_GOLDEN
# names a checkpoint (skip with -SkipGpu); rebuild dawnrt.exe when rt.cc is
# newer; build gptrun.exe stamped with the release; copy dawnrt and its DLLs;
# write manifest.txt (commits, dependency hashes, file SHA-256s); zip it; run
# the zipped build's `version` as a smoke test.
param(
    [Parameter(Mandatory = $true)][string]$Release,
    [switch]$AllowDirty,
    [switch]$SkipGpu
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root

function Step($name) { Write-Host "== $name" -ForegroundColor Cyan }
function Run($what, [scriptblock]$block) {
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}

# Sibling checkouts. hodgemix is https://github.com/EncryptidSystems/hodgemix.
$hodgemix = Join-Path (Split-Path $root) 'hodgemix-go'
$embeddb = Join-Path (Split-Path $root) 'EmbedDB'

Step 'source state'
$dirty = git status --porcelain
$hmDirty = git -C $hodgemix status --porcelain
if (-not $AllowDirty) {
    if ($dirty) { throw "ModelRun has uncommitted changes; commit them or pass -AllowDirty" }
    if ($hmDirty) { throw "hodgemix-go has uncommitted changes; commit them or pass -AllowDirty" }
}

Step 'vet and unit tests'
Run 'go vet' { go vet ./... }
Run 'go test' { go test -count=1 ./... }

if (-not $SkipGpu -and $env:MODELRUN_GOLDEN) {
    Step "GPU tests on $env:MODELRUN_GOLDEN"
    $tests = 'TestCacheShared|TestShelf|TestSampledExact|TestAttnExact'
    if ($env:MODELRUN_GOLDEN_FILE) { $tests += '|TestGolden' }
    Run 'GPU tests' { go test -count=1 -timeout 60m -run $tests ./internal/hodge }
} elseif (-not $SkipGpu) {
    throw "set MODELRUN_GOLDEN to a checkpoint .yaml for the GPU tests, or pass -SkipGpu"
}

Step 'dawnrt'
$rtExe = Join-Path $root 'dawnrt\dawnrt.exe'
$rtSrc = Join-Path $root 'dawnrt\rt.cc'
if (-not (Test-Path $rtExe) -or (Get-Item $rtSrc).LastWriteTime -gt (Get-Item $rtExe).LastWriteTime) {
    Run 'dawnrt build' { cmd /c (Join-Path $root 'dawnrt\build.bat') }
}

Step 'build'
$name = "gptrun-$Release"
$dist = Join-Path $root "dist\$name"
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Force (Join-Path $dist 'dawnrt') | Out-Null
Run 'go build' { go build -trimpath -ldflags "-X gptrun/internal/version.Release=$Release" -o (Join-Path $dist 'gptrun.exe') ./cmd/gptrun }
foreach ($f in 'dawnrt.exe', 'webgpu_dawn.dll', 'dxcompiler.dll', 'dxil.dll') {
    Copy-Item (Join-Path $root "dawnrt\$f") (Join-Path $dist "dawnrt\$f")
}
Copy-Item (Join-Path $root 'README.md') $dist

Step 'manifest'
function TreeHash($dir) {
    if (-not (Test-Path $dir)) { return 'missing' }
    $files = Get-ChildItem -Recurse -File $dir -Include *.go, go.mod, go.sum |
        Where-Object { $_.FullName -notmatch '\\(\.git|vendor)\\' } | Sort-Object FullName
    $sb = [System.Text.StringBuilder]::new()
    foreach ($f in $files) {
        $rel = $f.FullName.Substring($dir.Length)
        [void]$sb.AppendLine("$rel $((Get-FileHash $f.FullName -Algorithm SHA256).Hash)")
    }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($sb.ToString())
    $sha = [System.Security.Cryptography.SHA256]::Create()
    return ([System.BitConverter]::ToString($sha.ComputeHash($bytes)) -replace '-', '').ToLower()
}
$exe = Join-Path $dist 'gptrun.exe'
$lines = @(
    "release    $Release"
    "version    $(& $exe version)"
    "built      $((Get-Date).ToString('u'))"
    "go         $(go env GOVERSION)"
    "gptrun   $(git rev-parse HEAD)$(if ($dirty) { ' +dirty' })"
    "hodgemix   $(git -C $hodgemix rev-parse HEAD)$(if ($hmDirty) { ' +dirty' })  tree $(TreeHash $hodgemix)"
    "embeddb    tree $(TreeHash $embeddb)  (not a git repository)"
    ""
    "sha256"
)
Get-ChildItem -Recurse -File $dist | Sort-Object FullName | ForEach-Object {
    $lines += "  $((Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower())  $($_.FullName.Substring($dist.Length + 1))"
}
$lines | Set-Content -Encoding utf8 (Join-Path $dist 'manifest.txt')

Step 'zip'
$zip = Join-Path $root "dist\$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $dist -DestinationPath $zip

Step 'smoke test'
$check = Join-Path $root "dist\check-$Release"
if (Test-Path $check) { Remove-Item -Recurse -Force $check }
Expand-Archive $zip -DestinationPath $check
$v = & (Join-Path $check "$name\gptrun.exe") version
if ($LASTEXITCODE -ne 0 -or $v -notmatch [regex]::Escape($Release)) { throw "the zipped build says '$v'" }
Remove-Item -Recurse -Force $check
Write-Host "released $zip`n$v" -ForegroundColor Green
