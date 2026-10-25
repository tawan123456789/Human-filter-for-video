import shutil
from pathlib import Path

root = Path.cwd()
out = root / "dist" / "jetson_nano_256_gpu_bundle"
files = {
    root / "top2_benchmark" / "realtime_app.py": out / "top2_benchmark" / "realtime_app.py",
    root / "jetson_nano" / "stream_receiver.py": out / "jetson_nano" / "stream_receiver.py",
    root / "jetson_nano" / "webcam_to_json.py": out / "jetson_nano" / "webcam_to_json.py",
    root / "jetson_nano" / "webcam_to_jetson.py": out / "jetson_nano" / "webcam_to_jetson.py",
    root / "jetson_nano" / "nano256_gpu_bundle" / "README_TH.md": out / "README_TH.md",
    root / "jetson_nano" / "nano256_gpu_bundle" / "install.sh": out / "install.sh",
}
for src, dst in files.items():
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
shutil.copytree(root / "compression" / "outputs" / "jetson_nano_256",
                out / "compression" / "outputs" / "jetson_nano_256")
shutil.copytree(root / "jetson_nano" / "wheels", out / "wheels")
print(out)
