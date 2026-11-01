"""Native drop-in for ``v2d.docker.container.run_in_container``.

The upstream function builds a ``docker run ... python -m <module> --arg /data/<mount>/...``
command. This machine has no Docker, so this version runs ``python -m <module>`` in
the current venv with host paths instead.

Differences from the container path, handled here:
  * Paths are passed as host absolute paths (no /data/<arg> mounts).
  * ``env`` values that point into a container mount (e.g. FoundationPose's
    FOUNDATIONPOSE_WEIGHTS_DIR=/data/weights_dir/...) are mapped back to host paths.
  * ``extra_volumes`` of the form ``host:container[:ro]`` (SAM2 prompt masks) are
    honoured by rewriting container paths inside JSON input files to host paths.
  * Anonymous ``extra_volumes`` (dev-mode build preservation) are ignored.
  * Per-image environment that the Dockerfiles set is applied (see IMAGE_ENV).

Only images rebuilt natively are allowed. For any other image, point
``V2D_NATIVE_PYTHON_<IMAGE_NAME_UPPER>`` at a python interpreter that has that
module installed (e.g. a separate venv), otherwise a RuntimeError is raised.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_MODELS = os.environ.get("V2D_NATIVE_MODELS_ROOT", "/data/models")

# Images whose lib modules are installed in this venv, with the env their
# Dockerfiles set.
IMAGE_ENV: dict[str, dict[str, str]] = {
    "v2d_hamer": {"PYOPENGL_PLATFORM": "egl"},
    # HF_HUB_OFFLINE stops wilor-mini from silently fetching an unofficial
    # MANO_RIGHT.pkl from Hugging Face when the real one is missing.
    "v2d_wilor": {"PYOPENGL_PLATFORM": "egl", "HF_HUB_OFFLINE": "1"},
    "v2d_foundation_pose": {"OPENCV_IO_ENABLE_OPENEXR": "1"},
    "v2d_sam2": {},
    "v2d_grounding_dino": {
        "HF_HOME": f"{_MODELS}/grounding_dino/hf_home",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    },
    "v2d_moge": {},
    "v2d_anycalib": {},
}


def _base_dir(path: str) -> str:
    """Same rule as upstream: deepest directory without glob characters."""
    parts = Path(path).parts
    clean = []
    for part in parts[:-1]:
        if "*" in part or "?" in part:
            break
        clean.append(part)
    return str(Path(*clean)) if clean else str(Path(path).parent)


def _python_for(image: str) -> str:
    override = os.environ.get(f"V2D_NATIVE_PYTHON_{image.upper()}")
    if override:
        return override
    if image in IMAGE_ENV:
        return sys.executable
    raise RuntimeError(
        f"Docker image {image!r} has no native build in this env "
        f"(native: {sorted(IMAGE_ENV)}). Set V2D_NATIVE_PYTHON_{image.upper()} "
        "to an interpreter that provides it, or skip this pipeline stage."
    )


def _map_prefix(value: str, mapping: dict[str, str]) -> str:
    """Replace the longest matching container prefix in ``value`` with its host dir."""
    for cdir in sorted(mapping, key=len, reverse=True):
        if value == cdir or value.startswith(cdir + "/"):
            return mapping[cdir] + value[len(cdir):]
    return value


def _rewrite_json_paths(path: str, mapping: dict[str, str], tmpdir: str) -> str:
    """Copy a JSON input with container paths rewritten to host paths."""
    if not mapping or not path.endswith(".json") or not os.path.isfile(path):
        return path
    text = Path(path).read_text()
    if not any(c in text for c in mapping):
        return path

    def fix(obj):
        if isinstance(obj, dict):
            return {k: fix(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [fix(v) for v in obj]
        if isinstance(obj, str):
            return _map_prefix(obj, mapping)
        return obj

    out = os.path.join(tmpdir, os.path.basename(path))
    with open(out, "w") as f:
        json.dump(fix(json.loads(text)), f, indent=2)
    return out


def run_in_container(
    image: str,
    module: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    extra_args: dict[str, object] | None = None,
    dev: bool = False,
    modules_dir: str | None = None,
    gpus: bool = False,
    env: dict[str, str] | None = None,
    extra_volumes: list[str] | None = None,
) -> None:
    """Run ``python -m module`` natively with the same CLI the container would get."""
    del dev, modules_dir, gpus  # editable installs; GPU is always visible
    python = _python_for(image)

    inputs = {k: os.path.abspath(v) for k, v in inputs.items() if v is not None}
    outputs = {k: os.path.abspath(v) for k, v in outputs.items() if v is not None}
    for path in outputs.values():
        os.makedirs(_base_dir(path), exist_ok=True)

    # Rebuild the container mount table so env values written against it
    # (e.g. /data/weights_dir/<name>) can be mapped back to host paths.
    container_to_host: dict[str, str] = {}
    seen_dirs: set[str] = set()
    for arg_name, path in {**inputs, **outputs}.items():
        host_dir = _base_dir(path)
        if host_dir not in seen_dirs:
            seen_dirs.add(host_dir)
            container_to_host[f"/data/{arg_name}"] = host_dir

    volume_map: dict[str, str] = {}
    for vol in extra_volumes or []:
        parts = vol.split(":")
        if len(parts) >= 2 and parts[0].startswith("/"):
            volume_map[parts[1]] = parts[0]
        # bare paths are anonymous dev-preserve volumes: nothing to do natively

    tmpdir = tempfile.mkdtemp(prefix="v2d_native_")
    try:
        cmd = [python, "-m", module]
        for arg_name, path in {**inputs, **outputs}.items():
            if arg_name in inputs:
                path = _rewrite_json_paths(path, volume_map, tmpdir)
            cmd += [f"--{arg_name}", path]
        for arg_name, value in (extra_args or {}).items():
            if value is None or value is False:
                continue
            if value is True:
                cmd += [f"--{arg_name}"]
            else:
                cmd += [f"--{arg_name}", str(value)]

        run_env = dict(os.environ)
        run_env.update(IMAGE_ENV.get(image, {}))
        all_maps = {**container_to_host, **volume_map}
        for key, value in (env or {}).items():
            run_env[key] = _map_prefix(str(value), all_maps)
        run_env.setdefault("PYTHONUNBUFFERED", "1")

        print(f"[v2d-native] {image}: {' '.join(cmd[1:])}", flush=True)
        subprocess.run(cmd, check=True, env=run_env)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
