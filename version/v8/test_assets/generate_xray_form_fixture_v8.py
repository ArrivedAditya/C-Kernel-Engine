#!/usr/bin/env python3
"""Generate a deterministic large public form fixture without image dependencies."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


WIDTH = 1152
HEIGHT = 896


def generate(path: Path) -> str:
    image = bytearray([248, 248, 246]) * (WIDTH * HEIGHT)

    def pixel(x: int, y: int, rgb=(24, 28, 32)) -> None:
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            offset = (y * WIDTH + x) * 3
            image[offset:offset + 3] = bytes(rgb)

    def line(x0: int, y0: int, x1: int, y1: int, thickness=2) -> None:
        if y0 == y1:
            for y in range(y0, y0 + thickness):
                for x in range(x0, x1 + 1): pixel(x, y)
        elif x0 == x1:
            for x in range(x0, x0 + thickness):
                for y in range(y0, y1 + 1): pixel(x, y)

    def pseudo_text(x: int, y: int, text: str, scale=2) -> None:
        for index, value in enumerate(text.encode("ascii")):
            ox = x + index * 6 * scale
            for row in range(7):
                bits = ((value * 37 + row * 19) ^ (value >> (row % 3))) & 0x1F
                for col in range(5):
                    if bits & (1 << col):
                        for dy in range(scale):
                            for dx in range(scale): pixel(ox + col * scale + dx, y + row * scale + dy)

    line(40, 40, WIDTH - 40, 40, 4); line(40, HEIGHT - 40, WIDTH - 40, HEIGHT - 40, 4)
    line(40, 40, 40, HEIGHT - 40, 4); line(WIDTH - 40, 40, WIDTH - 40, HEIGHT - 40, 4)
    pseudo_text(76, 72, "PUBLIC SYNTHETIC SERVICE FORM", 3)
    pseudo_text(76, 125, "CASE ID: XRAY-2026-0711", 2)
    y = 180
    labels = ["FULL NAME", "ADDRESS", "CITY", "POSTAL CODE", "PHONE", "REQUEST TYPE", "DETAILS"]
    values = ["ALEX MORGAN", "123 PUBLIC TEST ROAD", "VICTORIA", "V8V 1A1", "250 555 0142", "INFORMATION", "SYNTHETIC DATA ONLY"]
    for label, value in zip(labels, values):
        pseudo_text(76, y, label, 2)
        line(330, y + 18, 1010, y + 18, 2)
        pseudo_text(350, y, value, 2)
        y += 76
    pseudo_text(76, 735, "CONSENT", 2)
    line(330, 726, 356, 726, 3); line(330, 726, 330, 752, 3); line(356, 726, 356, 752, 3); line(330, 752, 356, 752, 3)
    line(335, 738, 343, 748, 3); line(343, 748, 352, 731, 3)
    pseudo_text(378, 735, "YES", 2)
    payload = f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii") + bytes(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/xray/public_form_1152x896.ppm"))
    args = parser.parse_args()
    digest = generate(args.output)
    print(f"fixture={args.output} size={WIDTH}x{HEIGHT} sha256={digest}")


if __name__ == "__main__":
    main()
