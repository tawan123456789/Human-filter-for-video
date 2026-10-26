"""Repackage the existing ARM64 PyInstaller apps with the newly fine-tuned ONNX models."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BUNDLE = ROOT / "dist" / "jetson_nano_256_gpu_bundle.tar"
RUN = ROOT / "compression" / "outputs" / "final_fp16_augmented_20261026"
PACKAGE = ROOT / "jetson_nano" / "model_test_bundle"
PACKAGE_NAME = "jetson_nano_modelA_modelB_standalone_bundle"

APP_SPECS = (
    ("nano256_realtime/app/", "realtime_app/", "nano256_realtime", "realtime_app"),
    ("nano256_webcam_to_jetson/app/", "webcam_to_jetson/", "webcam_to_json", "webcam_to_jetson"),
)

MODEL_REPLACEMENTS = {
    "compression/outputs/jetson_nano_256/model_a_svd50/model_a_svd50_256_fp16.onnx": RUN / "exports" / "SVD50" / "svd50_fp16.onnx",
    "compression/outputs/jetson_nano_256/model_b_slim50_svd50/model_b_slim50_svd50_256_fp16.onnx": RUN / "exports" / "SLIM50_SVD50" / "slim50_svd50_fp16.onnx",
}


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> str:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = 0
    import io
    archive.addfile(info, io.BytesIO(data))
    return hashlib.sha256(data).hexdigest()


def add_file(archive: tarfile.TarFile, source: Path, destination: str, mode: int = 0o644) -> str:
    data = source.read_bytes()
    return add_bytes(archive, destination, data, mode)


def remap_member(member: tarfile.TarInfo, new_name: str) -> tarfile.TarInfo:
    info = copy.copy(member)
    info.name = new_name
    return info


def build(output: Path) -> None:
    if not SOURCE_BUNDLE.is_file():
        raise FileNotFoundError(f"Prebuilt ARM64 app bundle is missing: {SOURCE_BUNDLE}")
    for model in MODEL_REPLACEMENTS.values():
        if not model.is_file():
            raise FileNotFoundError(f"New ONNX model is missing: {model}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    checksums: dict[str, str] = {}
    prefix = PACKAGE_NAME + "/"
    with tarfile.open(SOURCE_BUNDLE, "r:") as source, tarfile.open(output, "w:") as target:
        for member in source.getmembers():
            if not member.isfile() and not member.isdir() and not member.issym() and not member.islnk():
                continue
            match = next((spec for spec in APP_SPECS if member.name.startswith("jetson_nano_256_gpu_bundle/" + spec[0])), None)
            if not match:
                continue
            source_prefix, target_prefix, old_executable, new_executable = match
            relative = member.name.split("jetson_nano_256_gpu_bundle/" + source_prefix, 1)[1]
            if relative.endswith("_256_fp32.onnx"):
                continue
            if relative == old_executable:
                relative = new_executable
            destination = prefix + target_prefix + relative
            model_key = relative.replace("\\", "/").removeprefix("_internal/")
            model_source = MODEL_REPLACEMENTS.get(model_key)
            if member.isfile() and model_source:
                data = model_source.read_bytes()
                info = remap_member(member, destination)
                info.size = len(data)
                import io
                target.addfile(info, io.BytesIO(data))
                checksums[destination[len(prefix):]] = hashlib.sha256(data).hexdigest()
                continue
            info = remap_member(member, destination)
            if member.isfile():
                fileobj = source.extractfile(member)
                if fileobj is None:
                    raise RuntimeError(f"Could not read archive member: {member.name}")
                target.addfile(info, fileobj)
                fileobj.close()
                if not relative.endswith(".onnx"):
                    fileobj = source.extractfile(member)
                    h = hashlib.sha256()
                    while chunk := fileobj.read(1024 * 1024):
                        h.update(chunk)
                    fileobj.close()
                    checksums[destination[len(prefix):]] = h.hexdigest()
            else:
                target.addfile(info)

        # Keep baseline as a comparison option; replace Model A/B with the new exports.
        baseline_source = "jetson_nano_256_gpu_bundle/compression/outputs/jetson_nano_256/baseline/baseline_256_fp16.onnx"
        baseline_member = source.getmember(baseline_source)
        baseline_stream = source.extractfile(baseline_member)
        baseline_data = baseline_stream.read()
        baseline_stream.close()
        baseline_dest = "compression/outputs/jetson_nano_256/baseline/baseline_256_fp16.onnx"
        checksums[baseline_dest] = add_bytes(target, prefix + baseline_dest, baseline_data)

        for relative, model_source in MODEL_REPLACEMENTS.items():
            checksums[relative] = add_file(target, model_source, prefix + relative)

        local_files = (
            (ROOT / "jetson_nano" / "webcam_to_jetson.py", "jetson_nano/webcam_to_jetson.py"),
            (ROOT / "top2_benchmark" / "realtime_app.py", "top2_benchmark/realtime_app.py"),
            (ROOT / "jetson_nano" / "stream_receiver.py", "jetson_nano/stream_receiver.py"),
            (PACKAGE / "README_STANDALONE_TH.md", "README_TH.md"),
            (PACKAGE / "MODEL_STATUS_TH.md", "MODEL_STATUS_TH.md"),
        )
        for path, relative in local_files:
            checksums[relative] = add_file(target, path, prefix + relative)

        manifest = json.dumps(checksums, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        add_bytes(target, prefix + "SHA256SUMS.json", manifest)

    print(f"Created {output} ({output.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / f"{PACKAGE_NAME}.tar")
    args = parser.parse_args()
    build(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
