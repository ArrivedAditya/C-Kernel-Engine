#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import shutil
import subprocess

FONT_PATHS = [
    Path('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf'),
    Path('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf'),
]


def _write_with_imagemagick(path: Path, width: int, height: int, lines: list[str], *, point_size: int = 42) -> bool:
    convert = shutil.which('convert') or shutil.which('magick')
    font = next((p for p in FONT_PATHS if p.exists()), None)
    if not convert or font is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    draw: list[str] = []
    y = 58
    for line in lines:
        safe = str(line).replace('\\', '\\\\').replace("'", "\\'")
        draw.append(f"text 34,{y} '{safe}'")
        y += int(point_size * 1.35)
    cmd = [
        convert,
        '-size', f'{width}x{height}',
        '-depth', '8',
        'xc:#fbfcfe',
        '-fill', '#ffffff', '-stroke', '#202838', '-strokewidth', '2',
        '-draw', f'rectangle 10,10 {width - 11},{height - 11}',
        '-font', str(font),
        '-pointsize', str(point_size),
        '-fill', '#111827', '-stroke', 'none',
        '-draw', ' '.join(draw),
        'ppm:' + str(path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


FONT = {
    " ": ["00000","00000","00000","00000","00000","00000","00000"],
    ".": ["00000","00000","00000","00000","00000","01100","01100"],
    ":": ["00000","01100","01100","00000","01100","01100","00000"],
    "-": ["00000","00000","00000","11111","00000","00000","00000"],
    "0": ["01110","10001","10011","10101","11001","10001","01110"],
    "1": ["00100","01100","00100","00100","00100","00100","01110"],
    "2": ["01110","10001","00001","00010","00100","01000","11111"],
    "3": ["11110","00001","00001","01110","00001","00001","11110"],
    "4": ["00010","00110","01010","10010","11111","00010","00010"],
    "5": ["11111","10000","11110","00001","00001","10001","01110"],
    "6": ["00110","01000","10000","11110","10001","10001","01110"],
    "7": ["11111","00001","00010","00100","01000","01000","01000"],
    "8": ["01110","10001","10001","01110","10001","10001","01110"],
    "9": ["01110","10001","10001","01111","00001","00010","01100"],
    "A": ["01110","10001","10001","11111","10001","10001","10001"],
    "B": ["11110","10001","10001","11110","10001","10001","11110"],
    "C": ["01110","10001","10000","10000","10000","10001","01110"],
    "D": ["11110","10001","10001","10001","10001","10001","11110"],
    "E": ["11111","10000","10000","11110","10000","10000","11111"],
    "F": ["11111","10000","10000","11110","10000","10000","10000"],
    "G": ["01110","10001","10000","10111","10001","10001","01110"],
    "H": ["10001","10001","10001","11111","10001","10001","10001"],
    "I": ["01110","00100","00100","00100","00100","00100","01110"],
    "J": ["00111","00010","00010","00010","10010","10010","01100"],
    "K": ["10001","10010","10100","11000","10100","10010","10001"],
    "L": ["10000","10000","10000","10000","10000","10000","11111"],
    "M": ["10001","11011","10101","10101","10001","10001","10001"],
    "N": ["10001","11001","10101","10011","10001","10001","10001"],
    "O": ["01110","10001","10001","10001","10001","10001","01110"],
    "P": ["11110","10001","10001","11110","10000","10000","10000"],
    "Q": ["01110","10001","10001","10001","10101","10010","01101"],
    "R": ["11110","10001","10001","11110","10100","10010","10001"],
    "S": ["01111","10000","10000","01110","00001","00001","11110"],
    "T": ["11111","00100","00100","00100","00100","00100","00100"],
    "U": ["10001","10001","10001","10001","10001","10001","01110"],
    "V": ["10001","10001","10001","10001","10001","01010","00100"],
    "W": ["10001","10001","10001","10101","10101","10101","01010"],
    "X": ["10001","10001","01010","00100","01010","10001","10001"],
    "Y": ["10001","10001","01010","00100","00100","00100","00100"],
    "Z": ["11111","00001","00010","00100","01000","10000","11111"],
}


def put_rect(img: list[bytearray], x0: int, y0: int, w: int, h: int, color: tuple[int, int, int]) -> None:
    height = len(img)
    width = len(img[0]) // 3
    r, g, b = color
    for y in range(max(0, y0), min(height, y0 + h)):
        row = img[y]
        for x in range(max(0, x0), min(width, x0 + w)):
            i = x * 3
            row[i:i + 3] = bytes((r, g, b))


def draw_text(img: list[bytearray], x: int, y: int, text: str, scale: int = 4, color: tuple[int, int, int] = (20, 24, 32)) -> None:
    cx = x
    for ch in text.upper():
        glyph = FONT.get(ch, FONT[" "])
        for gy, bits in enumerate(glyph):
            for gx, bit in enumerate(bits):
                if bit == "1":
                    put_rect(img, cx + gx * scale, y + gy * scale, scale, scale, color)
        cx += 6 * scale


def write_ppm(path: Path, width: int, height: int, lines: list[str], *, point_size: int = 42) -> None:
    if _write_with_imagemagick(path, width, height, lines, point_size=point_size):
        return
    img = [bytearray([248, 250, 252] * width) for _ in range(height)]
    put_rect(img, 8, 8, width - 16, height - 16, (255, 255, 255))
    put_rect(img, 8, 8, width - 16, 2, (36, 44, 56))
    put_rect(img, 8, height - 10, width - 16, 2, (36, 44, 56))
    put_rect(img, 8, 8, 2, height - 16, (36, 44, 56))
    put_rect(img, width - 10, 8, 2, height - 16, (36, 44, 56))
    y = 28
    for line in lines:
        draw_text(img, 28, y, line, scale=4)
        y += 42
    payload = b"".join(bytes(row) for row in img)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)


def main() -> int:
    out_dir = Path("version/v8/test_assets")
    write_ppm(out_dir / "v8_ocr_clean_text.ppm", 512, 256, ["CK OCR TEST", "TOTAL 42.50", "QWEN3 VL"], point_size=44)
    write_ppm(out_dir / "v8_ocr_table.ppm", 512, 256, ["ITEM  QTY  PRICE", "PEN   02   3.50", "TAX   00   0.42"], point_size=40)
    write_ppm(out_dir / "v8_ocr_receipt.ppm", 512, 256, ["STORE 17", "MILK  4.20", "BREAD 3.10", "TOTAL 7.30"], point_size=36)
    write_ppm(out_dir / "v8_ocr_paragraph.ppm", 768, 320, ["CPU INFERENCE RUNS", "WITHOUT A GPU", "CHECK THE OUTPUT TEXT", "AND REPORT ERRORS"], point_size=38)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
