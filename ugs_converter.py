#!/usr/bin/env python3
"""Converter for Discord Times .ugs unit sprite containers.

UGS layout used by the game:
    repeated records of <uint16 width><uint16 height><width*height uint16 pixels>

Pixels are transformed ARGB4444 values. The on-disk word is produced as
ROL16(ARGB4444, 3) XOR 0xAAAA. Fully transparent black therefore becomes
0xAAAA, but the format also supports 16 alpha levels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover - depends on the local Python installation
    raise SystemExit(
        "Не найдена библиотека Pillow. Установите её командой: python -m pip install Pillow"
    ) from exc


XOR_MASK = 0xAAAA
FRAME_NAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)


class UgsError(Exception):
    """A user-facing validation error."""


@dataclass(frozen=True)
class UgsInfo:
    frame_count: int
    width: int
    height: int
    file_size: int
    sha256: str


def rotate_left_16(value: int, count: int) -> int:
    count %= 16
    return ((value << count) | (value >> (16 - count))) & 0xFFFF


def rotate_right_16(value: int, count: int) -> int:
    count %= 16
    return ((value >> count) | (value << (16 - count))) & 0xFFFF


def decode_ugs_pixel(value: int) -> tuple[int, int, int, int]:
    argb4444 = rotate_right_16(value ^ XOR_MASK, 3)
    alpha4 = (argb4444 >> 12) & 0x0F
    red4 = (argb4444 >> 8) & 0x0F
    green4 = (argb4444 >> 4) & 0x0F
    blue4 = argb4444 & 0x0F
    # Bit replication is the exact 4-bit to 8-bit expansion: n * 17.
    return red4 * 17, green4 * 17, blue4 * 17, alpha4 * 17


def encode_ugs_pixel(red: int, green: int, blue: int, alpha: int) -> int:
    # The original Discord Times converter truncates each channel to its high nibble.
    argb4444 = ((alpha >> 4) << 12) | ((red >> 4) << 8) | ((green >> 4) << 4) | (blue >> 4)
    return rotate_left_16(argb4444, 3) ^ XOR_MASK


def read_ugs(path: Path) -> list[Image.Image]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise UgsError(f"Не удалось прочитать файл: {exc}") from exc

    if not data:
        raise UgsError("UGS-файл пуст.")

    frames: list[Image.Image] = []
    offset = 0
    expected_size: tuple[int, int] | None = None

    while offset < len(data):
        if len(data) - offset < 4:
            raise UgsError(f"Оборван заголовок кадра {len(frames)} по адресу {offset}.")
        width, height = struct.unpack_from("<HH", data, offset)
        if width == 0 or height == 0 or width > 4096 or height > 4096:
            raise UgsError(
                f"Некорректный размер кадра {len(frames)}: {width}×{height} "
                f"(адрес {offset})."
            )
        if expected_size is None:
            expected_size = (width, height)
        elif expected_size != (width, height):
            raise UgsError(
                f"Кадр {len(frames)} имеет размер {width}×{height}, "
                f"ожидался {expected_size[0]}×{expected_size[1]}."
            )

        pixel_bytes = width * height * 2
        record_end = offset + 4 + pixel_bytes
        if record_end > len(data):
            raise UgsError(
                f"Оборваны пиксели кадра {len(frames)}: требуется ещё "
                f"{record_end - len(data)} байт."
            )

        values = struct.unpack_from(f"<{width * height}H", data, offset + 4)
        image = Image.new("RGBA", (width, height))
        image.putdata([decode_ugs_pixel(value) for value in values])
        frames.append(image)
        offset = record_end

    return frames


def inspect_ugs(path: Path) -> UgsInfo:
    """Validate a UGS file and return a compact summary."""
    frames = read_ugs(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise UgsError(f"Не удалось прочитать файл: {exc}") from exc
    return UgsInfo(
        frame_count=len(frames),
        width=frames[0].width,
        height=frames[0].height,
        file_size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def create_backup(path: Path) -> Path:
    """Copy an existing file to the first available numbered .bak path."""
    if not path.is_file():
        raise UgsError(f"Нечего сохранять в резервную копию: {path}")

    candidate = path.with_name(path.name + ".bak")
    number = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak.{number}")
        number += 1
    try:
        shutil.copy2(path, candidate)
    except OSError as exc:
        raise UgsError(f"Не удалось создать резервную копию: {exc}") from exc
    return candidate


def open_in_file_manager(path: Path) -> None:
    """Open a directory (or select a file) using the platform file manager."""
    target = path if path.is_dir() else path.parent
    try:
        if sys.platform == "win32":
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except OSError as exc:
        raise UgsError(f"Не удалось открыть папку: {exc}") from exc


def make_preview(frames: list[Image.Image], columns: int = 8) -> Image.Image:
    width, height = frames[0].size
    rows = math.ceil(len(frames) / columns)
    preview = Image.new("RGBA", (columns * width, rows * height), (36, 36, 36, 255))

    # A subtle checkerboard makes transparent areas visible.
    draw = ImageDraw.Draw(preview)
    tile = 8
    for y in range(0, preview.height, tile):
        for x in range(0, preview.width, tile):
            shade = 52 if ((x // tile) + (y // tile)) % 2 else 68
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(shade, shade, shade, 255))

    for index, frame in enumerate(frames):
        preview.alpha_composite(frame, ((index % columns) * width, (index // columns) * height))
    return preview


def extract_ugs(source: Path, output_dir: Path) -> tuple[int, tuple[int, int]]:
    frames = read_ugs(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    digits = max(3, len(str(len(frames) - 1)))
    for index, frame in enumerate(frames):
        frame.save(output_dir / f"frame_{index:0{digits}d}.png")

    make_preview(frames).save(output_dir / "preview_sheet.png")
    metadata = {
        "format": "Discord Times UGS",
        "converter_format_version": 2,
        "pixel_format": "ROL16(ARGB4444, 3) XOR 0xAAAA, little-endian",
        "alpha_levels": 16,
        "transparent_black": "0xAAAA",
        "frame_count": len(frames),
        "width": frames[0].width,
        "height": frames[0].height,
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (output_dir / "ugs_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(frames), frames[0].size


def find_frames(input_dir: Path) -> list[Path]:
    numbered: list[tuple[int, Path]] = []
    try:
        entries = list(input_dir.iterdir())
    except OSError as exc:
        raise UgsError(f"Не удалось открыть папку с PNG: {exc}") from exc

    for path in entries:
        match = FRAME_NAME_RE.match(path.name)
        if match and path.is_file():
            numbered.append((int(match.group(1)), path))
    if not numbered:
        raise UgsError("Не найдены файлы frame_000.png, frame_001.png и т. д.")

    numbered.sort(key=lambda item: item[0])
    indices = [index for index, _ in numbered]
    expected = list(range(len(numbered)))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        raise UgsError(
            "Нумерация кадров должна идти подряд от нуля. "
            + (f"Пропущены: {missing}." if missing else f"Получены номера: {indices}.")
        )
    return [path for _, path in numbered]


def read_expected_metadata(input_dir: Path) -> dict[str, object] | None:
    metadata_path = input_dir / "ugs_info.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UgsError(f"Не удалось прочитать ugs_info.json: {exc}") from exc
    if not isinstance(metadata, dict):
        raise UgsError("ugs_info.json должен содержать объект JSON.")
    return metadata


def build_ugs(input_dir: Path, destination: Path, overwrite: bool = False) -> tuple[int, tuple[int, int]]:
    frame_paths = find_frames(input_dir)
    metadata = read_expected_metadata(input_dir)
    if metadata is not None and metadata.get("pixel_format") == "BGR565 little-endian":
        raise UgsError(
            "Эти PNG были распакованы старой версией конвертера с неверными цветами. "
            "Распакуйте исходный UGS заново в новую папку или удалите ugs_info.json, "
            "если PNG были полностью заменены вашими изображениями."
        )
    if metadata is not None and isinstance(metadata.get("frame_count"), int):
        expected_count = int(metadata["frame_count"])
        if len(frame_paths) != expected_count:
            raise UgsError(
                f"Количество кадров не совпадает с ugs_info.json: "
                f"найдено {len(frame_paths)}, ожидалось {expected_count}."
            )
    if destination.exists() and not overwrite:
        raise UgsError(f"Файл уже существует: {destination}. Используйте --force для замены.")

    output = bytearray()
    expected_size: tuple[int, int] | None = None

    for index, frame_path in enumerate(frame_paths):
        try:
            with Image.open(frame_path) as opened:
                frame = opened.convert("RGBA")
        except OSError as exc:
            raise UgsError(f"Не удалось прочитать {frame_path.name}: {exc}") from exc

        if expected_size is None:
            expected_size = frame.size
            if frame.width > 65535 or frame.height > 65535:
                raise UgsError("Размер кадра не помещается в формат UGS.")
            if metadata is not None:
                metadata_width = metadata.get("width")
                metadata_height = metadata.get("height")
                if isinstance(metadata_width, int) and isinstance(metadata_height, int):
                    if frame.size != (metadata_width, metadata_height):
                        raise UgsError(
                            f"Размер кадров не совпадает с ugs_info.json: "
                            f"получен {frame.width}x{frame.height}, "
                            f"ожидался {metadata_width}x{metadata_height}."
                        )
        elif frame.size != expected_size:
            raise UgsError(
                f"{frame_path.name}: размер {frame.width}×{frame.height}, "
                f"ожидался {expected_size[0]}×{expected_size[1]}."
            )

        output.extend(struct.pack("<HH", frame.width, frame.height))
        pixels = frame.get_flattened_data() if hasattr(frame, "get_flattened_data") else frame.getdata()
        for red, green, blue, alpha in pixels:
            value = encode_ugs_pixel(red, green, blue, alpha)
            output.extend(struct.pack("<H", value))

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
        with os.fdopen(fd, "wb") as temporary:
            temporary.write(output)
        os.replace(temporary_name, destination)
    except OSError as exc:
        try:
            os.unlink(temporary_name)
        except (OSError, UnboundLocalError):
            pass
        raise UgsError(f"Не удалось записать UGS: {exc}") from exc

    assert expected_size is not None
    return len(frame_paths), expected_size


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("Discord Times — UGS Converter")
    root.geometry("560x315")
    root.resizable(False, False)
    last_directory = Path.cwd()

    title = tk.Label(root, text="Конвертер спрайтов UGS", font=("Segoe UI", 16, "bold"))
    title.pack(pady=(22, 8))
    info = tk.Label(
        root,
        text=(
            "Распаковывает UGS в отдельные PNG и собирает их обратно.\n"
            "Цвет и 16 уровней прозрачности PNG сохраняются в игровом формате."
        ),
        justify="center",
        font=("Segoe UI", 10),
    )
    info.pack(pady=(0, 20))

    def remember(path: Path) -> None:
        nonlocal last_directory
        last_directory = path if path.is_dir() else path.parent

    def offer_to_open(path: Path, summary: str) -> None:
        if not messagebox.askyesno("Готово", summary + "\n\nОткрыть папку с результатом?"):
            return
        try:
            open_in_file_manager(path)
        except UgsError as exc:
            messagebox.showerror("Ошибка", str(exc))

    def extract_clicked() -> None:
        source_name = filedialog.askopenfilename(
            title="Выберите UGS",
            initialdir=last_directory,
            filetypes=[("UGS sprites", "*.ugs"), ("Все файлы", "*.*")],
        )
        if not source_name:
            return
        source = Path(source_name)
        remember(source)
        output_name = filedialog.askdirectory(
            title="Выберите папку для PNG",
            initialdir=last_directory,
        )
        if not output_name:
            return
        output = Path(output_name)
        remember(output)
        try:
            count, size = extract_ugs(source, output)
        except UgsError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return
        offer_to_open(
            output,
            f"Распаковано кадров: {count}\nРазмер: {size[0]}×{size[1]}\n\n{output_name}",
        )

    def build_clicked() -> None:
        input_name = filedialog.askdirectory(
            title="Выберите папку с frame_*.png",
            initialdir=last_directory,
        )
        if not input_name:
            return
        input_dir = Path(input_name)
        remember(input_dir)
        base_name = input_dir.name.removesuffix("_frames") or "sprites"
        destination_name = filedialog.asksaveasfilename(
            title="Сохранить UGS",
            initialdir=input_dir.parent,
            initialfile=f"{base_name}-new.ugs",
            defaultextension=".ugs",
            filetypes=[("UGS sprites", "*.ugs"), ("Все файлы", "*.*")],
        )
        if not destination_name:
            return
        destination = Path(destination_name)
        remember(destination)
        backup: Path | None = None
        if destination.exists() and not messagebox.askyesno("Подтверждение", f"Файл уже существует:\n{destination}\n\nЗаменить его?"):
            return
        try:
            if destination.exists():
                backup = create_backup(destination)
            count, size = build_ugs(input_dir, destination, overwrite=True)
        except UgsError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return
        backup_line = f"\nРезервная копия: {backup}" if backup else ""
        offer_to_open(
            destination,
            f"Собрано кадров: {count}\nРазмер: {size[0]}×{size[1]}\n"
            f"Файл: {destination}{backup_line}",
        )

    def inspect_clicked() -> None:
        source_name = filedialog.askopenfilename(
            title="Проверить UGS",
            initialdir=last_directory,
            filetypes=[("UGS sprites", "*.ugs"), ("Все файлы", "*.*")],
        )
        if not source_name:
            return
        source = Path(source_name)
        remember(source)
        try:
            details = inspect_ugs(source)
        except UgsError as exc:
            messagebox.showerror("UGS повреждён", str(exc))
            return
        messagebox.showinfo(
            "UGS исправен",
            f"Кадров: {details.frame_count}\n"
            f"Размер кадра: {details.width}×{details.height}\n"
            f"Размер файла: {details.file_size:,} байт\n"
            f"SHA-256: {details.sha256}",
        )

    buttons = tk.Frame(root)
    buttons.pack()
    tk.Button(buttons, text="Распаковать UGS → PNG", width=24, height=2, command=extract_clicked).grid(row=0, column=0, padx=8)
    tk.Button(buttons, text="Собрать PNG → UGS", width=24, height=2, command=build_clicked).grid(row=0, column=1, padx=8)
    tk.Button(root, text="Проверить UGS", width=24, command=inspect_clicked).pack(pady=(14, 0))
    tk.Label(root, text="PNG должны называться frame_000.png, frame_001.png…", fg="#555555").pack(pady=14)
    root.mainloop()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Конвертер спрайтов Discord Times UGS ↔ PNG")
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract", help="распаковать UGS в PNG")
    extract_parser.add_argument("source", type=Path, help="исходный .ugs")
    extract_parser.add_argument("output_dir", type=Path, help="папка для PNG")

    build_parser = subparsers.add_parser("build", help="собрать UGS из frame_*.png")
    build_parser.add_argument("input_dir", type=Path, help="папка с PNG")
    build_parser.add_argument("destination", type=Path, help="выходной .ugs")
    build_parser.add_argument("--force", action="store_true", help="разрешить замену существующего файла")

    inspect_parser = subparsers.add_parser("inspect", help="проверить UGS и показать сведения")
    inspect_parser.add_argument("source", type=Path, help="проверяемый .ugs")
    return parser


def main(argv: list[str] | None = None) -> int:
    # A legacy Windows console may not support every character used in messages.
    # Replacing an unsupported glyph is preferable to failing after conversion.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = create_parser().parse_args(argv)
    if args.command is None:
        run_gui()
        return 0
    try:
        if args.command == "extract":
            count, size = extract_ugs(args.source, args.output_dir)
            print(f"Готово: распаковано {count} кадров {size[0]}x{size[1]} в {args.output_dir}")
        elif args.command == "build":
            count, size = build_ugs(args.input_dir, args.destination, args.force)
            print(f"Готово: собрано {count} кадров {size[0]}x{size[1]} в {args.destination}")
        elif args.command == "inspect":
            details = inspect_ugs(args.source)
            print(
                f"UGS исправен: {details.frame_count} кадров "
                f"{details.width}x{details.height}, {details.file_size} байт\n"
                f"SHA-256: {details.sha256}"
            )
    except UgsError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
