import os
import tarfile

source = "dist/jetson_nano_512_gpu_bundle.tar"
temporary = "build/jetson_nano_512_gpu_bundle_single.tar"
with tarfile.open(source, "r") as old:
    latest = {}
    for member in old.getmembers():
        latest[member.name] = member
    with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as new:
        for member in latest.values():
            if member.isfile():
                stream = old.extractfile(member)
                new.addfile(member, stream)
                stream.close()
            else:
                new.addfile(member)
os.replace(temporary, source)
with tarfile.open(source, "r") as archive:
    members = archive.getmembers()
    paths = {m.name: m for m in members}
    executable = "jetson_nano_512_gpu_bundle/nano512_realtime/app/nano512_realtime"
    libpython = "jetson_nano_512_gpu_bundle/nano512_realtime/app/_internal/libpython3.8.so.1.0"
    app_source = "jetson_nano_512_gpu_bundle/top2_benchmark/realtime_app.py"
    print("members:", len(members), "unique:", len(paths))
    print("runtime lib:", paths[libpython].size, "bytes")
    print("app executable:", paths[executable].size, "bytes, mode", oct(paths[executable].mode))
    print("bad provider identifier in source:", b"chosen_provider" in archive.extractfile(app_source).read())
print("bundle bytes:", os.path.getsize(source))
