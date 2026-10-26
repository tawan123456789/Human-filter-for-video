$ErrorActionPreference = 'Stop'

$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ArchivePath = Join-Path $RepoDir 'dist/nano1_webcam_sender_opset16_gpu_bundle.tar'
$Dockerfile = Join-Path $PSScriptRoot 'Dockerfile.jetson_nano'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker Desktop is required to build the Linux ARM64 package.'
}
$ModelCandidates = @(
    (Join-Path $RepoDir 'jetson_nano/models/rvm_mobilenetv1.onnx'),
    (Join-Path $RepoDir 'jetson_nano/models/default_model.onnx'),
    (Join-Path $RepoDir 'compression/outputs/jetson_nano/SLIM50_SVD50.onnx')
)
if (-not ($ModelCandidates | Where-Object { Test-Path -LiteralPath $_ })) {
    throw 'No suitable ONNX model found. Run `python export_onnx.py` first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'wheels/onnxruntime_gpu-1.11.0-cp38-cp38-linux_aarch64.whl'))) {
    throw 'The Jetson ONNX Runtime GPU wheel is missing from jetson_nano/wheels/.'
}
if (Test-Path -LiteralPath $ArchivePath) {
    throw "Output already exists: $ArchivePath. Move or rename it before rebuilding."
}

Push-Location $RepoDir
try {
    docker buildx build `
      --platform linux/arm64 `
      --file $Dockerfile `
      --output "type=tar,dest=$ArchivePath" `
      .
    if ($LASTEXITCODE -ne 0) {
        throw "Docker ARM64 build failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Write-Host "Package created: $ArchivePath"
Write-Host 'Copy this tar archive to Nano 1 and extract it; the ONNX model is bundled inside.'
