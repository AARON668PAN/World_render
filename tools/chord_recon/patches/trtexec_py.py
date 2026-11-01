#!/usr/bin/env python
"""Minimal ``trtexec`` stand-in built on the TensorRT Python API.

pip's tensorrt-cu12 wheels ship no trtexec binary. FoundationPose's
trt_engine.build_engine() only calls:
  trtexec --onnx=X --saveEngine=Y --minShapes=a:NxCxHxW,b:... --optShapes=... --maxShapes=... --noTF32 --skipInference
so that is all this script supports. FP32 engine, TF32 disabled, one optimization profile.
"""
import sys

import tensorrt as trt


def _shapes(spec: str) -> dict[str, tuple[int, ...]]:
    out = {}
    for item in spec.split(","):
        name, dims = item.rsplit(":", 1)
        out[name] = tuple(int(d) for d in dims.split("x"))
    return out


def main(argv: list[str]) -> int:
    opts, flags = {}, set()
    for a in argv:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            opts[k] = v
        elif a.startswith("--"):
            flags.add(a[2:])
        else:
            raise SystemExit(f"trtexec_py: unsupported argument {a!r}")
    unknown = set(opts) - {"onnx", "saveEngine", "minShapes", "optShapes", "maxShapes"}
    unknown |= flags - {"noTF32", "skipInference"}
    if unknown:
        raise SystemExit(f"trtexec_py: unsupported options {sorted(unknown)}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is the only mode in TRT 10
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(opts["onnx"]):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        return 1
    config = builder.create_builder_config()
    if "noTF32" in flags:
        config.clear_flag(trt.BuilderFlag.TF32)
    if "minShapes" in opts:
        profile = builder.create_optimization_profile()
        mins, opts_, maxs = (_shapes(opts[k]) for k in ("minShapes", "optShapes", "maxShapes"))
        for name in mins:
            profile.set_shape(name, mins[name], opts_[name], maxs[name])
        config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("trtexec_py: engine build failed", file=sys.stderr)
        return 1
    with open(opts["saveEngine"], "wb") as f:
        f.write(engine)
    print(f"trtexec_py: wrote {opts['saveEngine']} (TensorRT {trt.__version__})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
