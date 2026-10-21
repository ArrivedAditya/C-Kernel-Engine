"""Capture observed convolution operands through the pinned MTMD callback."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
from pathlib import Path

from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loaded_mtmd_library() -> Path:
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        candidate = line.split()[-1]
        if Path(candidate).name.startswith("libmtmd.so"):
            path = Path(candidate).resolve()
            if path.is_file():
                return path
    raise RuntimeError("loaded MTMD library identity is unavailable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument("--projector", type=Path, required=True)
    parser.add_argument("--oracle-source", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conv-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.conv_index < 0 or args.threads < 1:
        parser.error("convolution index and thread count must be nonnegative/positive")
    if any(not path.is_file() for path in (args.shim, args.projector, args.image)):
        parser.error("shim, projector and image files are required")
    if not args.oracle_source.is_dir():
        parser.error("pinned oracle source directory is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("capture directory must be empty to prevent stale evidence")
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output_dir.chmod(0o700)
    os.environ["CK_LLAMA_PARITY_DIR"] = str(args.output_dir.resolve())
    os.environ["CK_LLAMA_PARITY_CONV_INDEX"] = str(args.conv_index)
    os.environ["CK_LLAMA_PARITY_META"] = "1"
    library = ctypes.CDLL(str(args.shim.resolve()))
    mtmd_library = _loaded_mtmd_library()
    revision = subprocess.run(
        ["git", "-C", str(args.oracle_source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != args.expected_revision:
        raise RuntimeError("oracle source revision differs from the required pin")
    dirty = subprocess.run(
        ["git", "-C", str(args.oracle_source), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("pinned oracle source has tracked modifications")
    library.ck_mtmd_clip_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_int]
    library.ck_mtmd_clip_init.restype = ctypes.c_void_p
    library.ck_mtmd_clip_free.argtypes = [ctypes.c_void_p]
    library.ck_mtmd_clip_preprocess_rgb8.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    library.ck_mtmd_clip_preprocess_rgb8.restype = ctypes.c_int64
    library.ck_mtmd_clip_embd_nbytes_by_img.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    library.ck_mtmd_clip_embd_nbytes_by_img.restype = ctypes.c_size_t
    library.ck_mtmd_clip_encode_float_image.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ]
    library.ck_mtmd_clip_encode_float_image.restype = ctypes.c_int
    with Image.open(args.image) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        source = (ctypes.c_uint8 * (width * height * 3)).from_buffer_copy(rgb.tobytes())
    handle = library.ck_mtmd_clip_init(str(args.projector.resolve()).encode(), 0, 0, 0, 0, 0)
    if not handle:
        raise RuntimeError("pinned MTMD projector initialization failed")
    try:
        out_width = ctypes.c_int()
        out_height = ctypes.c_int()
        count = library.ck_mtmd_clip_preprocess_rgb8(
            handle, source, width, height, None, 0,
            ctypes.byref(out_width), ctypes.byref(out_height),
        )
        if count <= 0 or count > (1 << 27):
            raise RuntimeError("MTMD preprocessing extent is invalid")
        processed = (ctypes.c_float * count)()
        actual = library.ck_mtmd_clip_preprocess_rgb8(
            handle, source, width, height, processed, count,
            ctypes.byref(out_width), ctypes.byref(out_height),
        )
        if actual != count or out_width.value <= 0 or out_height.value <= 0:
            raise RuntimeError("MTMD preprocessing changed extent between calls")
        processed_sha256 = hashlib.sha256(ctypes.string_at(processed, count * 4)).hexdigest()
        output_bytes = library.ck_mtmd_clip_embd_nbytes_by_img(handle, out_width.value,
                                                                out_height.value)
        if output_bytes <= 0 or output_bytes > (1 << 30) or output_bytes % 4:
            raise RuntimeError("MTMD embedding extent is invalid")
        output = (ctypes.c_float * (output_bytes // 4))()
        if not library.ck_mtmd_clip_encode_float_image(
            handle, args.threads, processed, out_height.value, out_width.value, output,
        ):
            raise RuntimeError("MTMD encoder execution failed")
    finally:
        library.ck_mtmd_clip_free(handle)
    if not (args.output_dir / "meta.jsonl").is_file():
        raise RuntimeError("oracle did not produce convolution capture metadata")
    selected = [json.loads(line) for line in (args.output_dir / "meta.jsonl").read_text().splitlines()
                if line.strip()]
    if not any(row.get("capture_id") == args.conv_index and
               row.get("role") == "conv_im2col" and row.get("status") == "observed"
               for row in selected):
        raise RuntimeError("selected convolution was not observed")
    identity = {
        "image_path": str(args.image.resolve()),
        "image_sha256": _sha256(args.image),
        "projector_path": str(args.projector.resolve()),
        "projector_sha256": _sha256(args.projector),
        "shim_path": str(args.shim.resolve()),
        "shim_sha256": _sha256(args.shim),
        "loaded_mtmd_sha256": _sha256(mtmd_library),
        "loaded_mtmd_path": str(mtmd_library),
        "oracle_source_revision": revision,
        "processed_input_sha256": processed_sha256,
        "processed_width": out_width.value,
        "processed_height": out_height.value,
        "selected_capture_id": args.conv_index,
        "threads": args.threads,
        "backend": "cpu",
    }
    destination = args.output_dir / "capture-identity.json"
    destination.write_text(json.dumps(identity, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    destination.chmod(0o600)
    print("Observed MTMD convolution capture complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
