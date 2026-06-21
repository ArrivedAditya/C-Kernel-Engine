#!/usr/bin/env python3
from __future__ import annotations

"""Convert safetensors to a RAM-only memfd BUMP and optionally run a command.

This is for very large safetensors checkpoints where there is enough RAM but not
enough disk or /dev/shm capacity to materialize weights.bump.  The script creates
an anonymous Linux memfd, invokes the v8 safetensors converter with the inherited
fd path, then keeps that fd alive while running a child command.  Use the literal
``{weights}`` in the child command; it is replaced with ``/proc/self/fd/<fd>``.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CONVERTER = SCRIPT_DIR / "convert_safetensors_to_bump_v8.py"


def _memfd_create(name: str) -> int:
    if not hasattr(os, "memfd_create"):
        raise SystemExit("os.memfd_create is not available on this Python/Linux build")
    fd = os.memfd_create(name, flags=0)
    os.set_inheritable(fd, True)
    return fd


def _run(cmd: list[str], *, pass_fd: int) -> int:
    proc = subprocess.run(cmd, pass_fds=(pass_fd,), check=False)
    return int(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--config-out", required=True, type=Path)
    ap.add_argument("--manifest-out", required=True, type=Path)
    ap.add_argument("--audit-out", type=Path)
    ap.add_argument("--arch", default="auto")
    ap.add_argument("--dtype", default="preserve", choices=("preserve", "bf16", "fp32"))
    ap.add_argument("--config-template", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="run converter dry-run without creating a memfd")
    ap.add_argument("--keep-open", action="store_true", help="after conversion, print fd path and wait for Enter; useful for manual inspection")
    ap.add_argument("command", nargs=argparse.REMAINDER, help="optional command after --; replace {weights} with the memfd path")
    args = ap.parse_args()

    if args.dry_run:
        cmd = [
            sys.executable,
            str(CONVERTER),
            "--checkpoint", str(args.checkpoint),
            "--output", "weights.bump",
            "--config-out", str(args.config_out),
            "--manifest-out", str(args.manifest_out),
            "--arch", str(args.arch),
            "--dtype", str(args.dtype),
            "--dry-run",
        ]
        if args.audit_out:
            cmd.extend(["--audit-out", str(args.audit_out)])
        if args.config_template:
            cmd.extend(["--config-template", str(args.config_template)])
        return subprocess.run(cmd, check=False).returncode

    fd = _memfd_create("ck-v8-weights.bump")
    fd_path = f"/proc/self/fd/{fd}"
    try:
        convert_cmd = [
            sys.executable,
            str(CONVERTER),
            "--checkpoint", str(args.checkpoint),
            "--output", fd_path,
            "--config-out", str(args.config_out),
            "--manifest-out", str(args.manifest_out),
            "--arch", str(args.arch),
            "--dtype", str(args.dtype),
        ]
        if args.audit_out:
            convert_cmd.extend(["--audit-out", str(args.audit_out)])
        if args.config_template:
            convert_cmd.extend(["--config-template", str(args.config_template)])

        print(f"[memfd-bump] converting into anonymous RAM fd {fd_path}")
        rc = _run(convert_cmd, pass_fd=fd)
        if rc != 0:
            return rc
        size = os.lseek(fd, 0, os.SEEK_END)
        os.lseek(fd, 0, os.SEEK_SET)
        print(f"[memfd-bump] ready path={fd_path} size={size / 1024 / 1024 / 1024:.2f} GiB")

        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        if command:
            child_cmd = [fd_path if item == "{weights}" else item for item in command]
            print("[memfd-bump] running:", " ".join(child_cmd))
            return _run(child_cmd, pass_fd=fd)
        if args.keep_open:
            print(f"[memfd-bump] fd is open at {fd_path}; press Enter to close and release RAM")
            try:
                input()
            except EOFError:
                pass
        else:
            print("[memfd-bump] no command provided; closing fd now releases the RAM BUMP")
        return 0
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
